import asyncio
from threading import Thread
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
from mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest
import requests

global consulta;


#Necesarios para el envio e inicio del contador
consulta = 'http://152.173.152.107:3333/acumulador';
t0 = time.time()
aforo = 0

#Daemon para el envio de informacion al backend
def envio():
    anterior = 0
    while True:
        if (aforo == 0):
            print('envia aforo',aforo)
            myobj = {'lab': 1, 'value': aforo}
            r = requests.post(consulta, data=myobj)
            print('respuesta desde la API', r)
        elif (anterior != aforo):
            print('envia aforo', aforo)
            myobj = {'lab': 1, 'value': aforo}
            r = requests.post(consulta, data=myobj)
            print('respuesta desde la API', r)
            anterior = aforo;
        time.sleep(5)

def run():
    # paso de argumentos para los modelos pre entranados caffe
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=False,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str,
                    help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
                    help="path to optional output video file")
    # confidence default 0.4
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
                    help="# of skip frames between detections")
    args = vars(ap.parse_args())

    # Lista de labels que SSD MobileNet está entrenado para detectar
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # Carga del modelo
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # if a video path was not supplied, grab a reference to the ip camera
    if not args.get("input", False):
        print("[INFO] Starting the live stream..")
        vs = VideoStream(config.url).start()
        time.sleep(2.0)

    # otherwise, grab a reference to the video file
    else:
        print("[INFO] Starting the video..")
        vs = cv2.VideoCapture(args["input"])

    # Inicializando variables, writer para video, W y H para el primer frame
    writer = None
    W = None
    H = None

    #declaración de variable aforo
    #Declaración de hilo como daemon e inicio
    global aforo

    threadEnvio = Thread(target = envio)
    threadEnvio.daemon = True;
    threadEnvio.start()

    #Instanciando el centroidTracker
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0
    x = []
    empty = []
    empty1 = []

    # start the frames per second throughput estimator
    fps = FPS().start()

    if config.Thread:
        print('hilo activado');
    # vs = thread.ThreadingClass(config.url)

    # Inicia lectura de frames
    while True:
        # Toma el siguiente frame de video ó stream
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame

        # Termina el proceso si se termina el video
        if args["input"] is not None and frame is None:
            break

        # Redimensiona frame a un máximo de 500px y luego convierte de BGR a RGB
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Si el tamaño del frame no está definido, se define.
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # Inicializa el grabado de video
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                     (W, H), True)

        # Inicia el estado actual del detector
        status = "Esperando.."
        rects = []

        # Chequea si potenciar el detector para el trackeo
        if totalFrames % args["skip_frames"] == 0:

            status = "Detecting"
            trackers = []

            # Convierte el frame a BLOB, la pasa a la red y obtiene la detección
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # Inicia el loop para la detección
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # Filtrado de detecciones, minimo 'confidence'
                if confidence > args["confidence"]:

                    # Extrae el indice del label de la clase de la lista de detecciones
                    idx = int(detections[0, 0, i, 1])

                    # Si el label de la clase no es 'Persona', ignorar.
                    if CLASSES[idx] != "person":
                        continue

                    # Calculo de los x,y del cuadro del objeto.
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Crea rectangulo en el objeto e inicia trackeo
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # Agregando el objeto trackeado a la lista de trackeados.
                    trackers.append(tracker)

        # Se inicia el trackeo de objetos en lugar de la deteccion de objetos, de modo de obtener un mejor
        # performance en procesamiento de frames.
        else:
            # recorre los objetos trackeados
            for tracker in trackers:
                status = "Tracking"

                # Actualiza objeto trackeado y actualiza posición.
                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))

        # Dibuja la linea horizontal en el centro del frame
        # De modo de que si un objeto 'Person' cruza esta linea determina si entra o sale
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
        cv2.putText(frame, "-Borde de prediccion - Entrada-", (10, H - ((i * 20) + 150)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Usa el tracker de centroides para asociar el centroide del objeto anterior
        # con el nuevo centroide calculado para el objeto
        objects = ct.update(rects)

        # Loop de los objetos trackeados.
        for (objectID, centroid) in objects.items():

            # Checkea si existe un objeto trackeable para el actual ID de objeto.
            to = trackableObjects.get(objectID, None)

            # Si no existe un objeto trackeable, se crea.
            if to is None:
                to = TrackableObject(objectID, centroid)

            else:
                # Con la resta del Y anterior y el Y actual, se determina la dirección
                # en la que se mueve el objeto, Negativo para salir, positivo para entrar.
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # Checkea si el objeto no ha sido contado.
                if not to.counted:

                    # Si direction es negativa, va saliend y el centroide está cerca
                    # de la linea central, luego cuenta el objeto.
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1

                        empty.append(totalUp)
                        to.counted = True

                    # Si direction es positiva, va entrando y el centroide está cerca
                    # de la linea central, luego cuenta el objeto.
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        empty1.append(totalDown)

                        # Si se excede el limite de personas, envía alerta.
                        if sum(x) >= config.Threshold:
                            cv2.putText(frame, "-ALERTA: Limite de aforo excedido-", (10, frame.shape[0] - 80),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                            if config.ALERT:
                                print("[INFO] Enviando alerta..")
                                Mailer().send(config.MAIL)
                                print("[INFO] Alerta enviada")

                        to.counted = True

                    x = []
                    # Cálculo del total de personas en el interior del lab
                    x.append(len(empty1) - len(empty))
                    aforo = len(empty1) - len(empty)

            # Guarda el objeto en el diccionario.
            trackableObjects[objectID] = to

            # Dibuja el ID y el centroide del objeto
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # Información a mostrar.
        info = [
            ("Salieron", totalUp),
            ("Entraron", totalDown),
            ("Estado", status),
        ]

        info2 = [
            ("Personas en sala", x)
        ]
        #asdasdsd
        # Muestra el frame.
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Initiate a simple log to save data at end of the day
        if config.Log:
            datetimee = [datetime.datetime.now()]
            d = [datetimee, empty1, empty, x]
            export_data = zip_longest(*d, fillvalue='')

            with open('Log.csv', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(("End Time", "In", "Out", "Total Inside"))
                wr.writerows(export_data)

        # Checkea si se debe guardar el frame en el pc.
        if writer is not None:
            writer.write(frame)

        # Muestra el frame
        cv2.imshow("- / S M C A / - Lab1", frame)
        key = cv2.waitKey(1) & 0xFF

        # Q para terminar proceso.
        if key == ord("q"):
            break

        # Aumenta el total de frames procesados y actualiza el contador de FPS
        totalFrames += 1
        fps.update()

        if config.Timer:
            # Automatic timer to stop the live stream. Set to 8 hours (28800s).
            t1 = time.time()
            num_seconds = (t1 - t0)
            if num_seconds > 50000:
                break

    # Detiene el timer y muestra información
    fps.stop()
    print("[INFO] Tiempo transcurrido: {:.2f}".format(fps.elapsed()))
    print("[INFO] FPS aprox.: {:.2f}".format(fps.fps()))

    # # if we are not using a video file, stop the camera video stream
    # if not args.get("input", False):
    # 	vs.stop()
    #
    # # otherwise, release the video file pointer
    # else:
    # 	vs.release()

    # issue 15
    if config.Thread:
        vs.release()

    # close any open windows
    cv2.destroyAllWindows()


##//Configuración de horario de inicio automático//

##learn more about different schedules here: https://pypi.org/project/schedule/
if config.Scheduler:
    ##Runs for every 1 second
    # schedule.every(1).seconds.do(run)
    ##Runs at every day (9:00 am). You can change it.
    schedule.every().day.at("9:00").do(run)

    while 1:
        schedule.run_pending()

else:
    run()
