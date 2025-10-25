"""
Servicio para reconocimiento facial con OpenCV
"""
import cv2
import os
import numpy as np
import time
import base64
import re
from django.conf import settings
from .asistencia_service import asistencia_service


class ReconocimientoService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ReconocimientoService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Inicializa los modelos de OpenCV"""
        self.data_path = settings.DATA_PATH
        self.model_path = settings.MODEL_PATH
        
        # Crear directorio Data si no existe
        os.makedirs(self.data_path, exist_ok=True)
        
        # Clasificador de rostros
        self.face_classif = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Reconocedor facial
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Cargar modelo si existe
        if os.path.exists(self.model_path):
            self.face_recognizer.read(str(self.model_path))
            self.image_paths = os.listdir(self.data_path)
            self.label_dict = {name: idx for idx, name in enumerate(self.image_paths)}
            self.next_label = len(self.image_paths)
        else:
            self.image_paths = []
            self.label_dict = {}
            self.next_label = 0
        
        # Control de asistencias
        self.estudiantes_reconocidos = set()
        self.tiempos_reconocimiento = {}
        self.duracion_reconocimiento = settings.RECONOCIMIENTO_CONFIG['duracion_reconocimiento']
        
        print(f"Modelo cargado. Personas: {self.image_paths}")
        print(f"Label dict: {self.label_dict}, Next label: {self.next_label}")

    def decode_image(self, data_url):
        """
        Decodifica una imagen base64 a numpy array
        
        Args:
            data_url: String con imagen en base64 (data:image/...)
            
        Returns:
            numpy.ndarray: Imagen en formato BGR
        """
        b64 = re.sub(r'^data:image/.+;base64,', '', data_url)
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def detectar_rostro(self, frame):
        """
        Detecta rostros en una imagen
        
        Args:
            frame: Imagen BGR
            
        Returns:
            tuple: (gray_frame, faces) donde faces es lista de (x,y,w,h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classif.detectMultiScale(gray, 1.3, 5)
        return gray, faces

    def reconocer_rostro(self, image_data):
        """
        Reconoce un rostro en una imagen base64
        
        Args:
            image_data: Imagen en base64
            
        Returns:
            dict: Resultado del reconocimiento
        """
        frame = self.decode_image(image_data)
        gray, faces = self.detectar_rostro(frame)
        
        if len(faces) == 0:
            return {"estado": "sin_rostro"}
        
        # Procesar solo el primer rostro
        x, y, w, h = faces[0]
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        
        label, confianza = self.face_recognizer.predict(rostro)
        
        box = [int(x), int(y), int(w), int(h)]
        
        if confianza < settings.RECONOCIMIENTO_CONFIG['confianza_threshold'] and label < len(self.image_paths):
            nombre = self.image_paths[label]
            
            # Lógica de registro único con tiempo
            if nombre not in self.tiempos_reconocimiento:
                self.tiempos_reconocimiento[nombre] = time.time()
            elif time.time() - self.tiempos_reconocimiento[nombre] >= self.duracion_reconocimiento:
                if nombre not in self.estudiantes_reconocidos:
                    self.estudiantes_reconocidos.add(nombre)
                    # Registrar asistencia en Firebase
                    asistencia_service.registrar_asistencia(nombre)
            
            return {
                "estado": "reconocido",
                "estudiante": nombre,
                "confianza": float(confianza),
                "box": box
            }
        else:
            return {
                "estado": "desconocido",
                "confianza": float(confianza),
                "box": box
            }

    def guardar_foto_registro(self, estudiante, foto_base64):
        """
        Guarda una foto de registro detectando y recortando el rostro
        
        Args:
            estudiante: Nombre del estudiante
            foto_base64: Foto en base64
            
        Returns:
            dict: {"ok": bool, "msg": str, "ruta": str (opcional)}
        """
        # Sanear nombre
        estudiante = os.path.splitext(os.path.basename(estudiante))[0]
        person_path = os.path.join(self.data_path, estudiante)
        os.makedirs(person_path, exist_ok=True)
        
        # Decodificar imagen
        try:
            header, encoded = foto_base64.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return {"ok": False, "msg": f"Error decodificando imagen: {str(e)}"}
        
        # Detectar rostro
        gray, faces = self.detectar_rostro(img)
        
        if len(faces) == 0:
            return {"ok": False, "msg": "no face"}
        
        # Recortar primer rostro
        x, y, w, h = faces[0]
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        
        # Guardar
        timestamp = int(time.time() * 1000)
        ruta = os.path.join(person_path, f'rostro_{timestamp}.jpg')
        cv2.imwrite(ruta, rostro)
        
        return {"ok": True, "msg": "guardado", "ruta": ruta}

    def entrenar_incremental(self, estudiante):
        """
        Entrena el modelo de forma incremental con las fotos de un estudiante
        
        Args:
            estudiante: Nombre del estudiante
            
        Returns:
            dict: Resultado del entrenamiento
        """
        person_path = os.path.join(self.data_path, estudiante)
        
        if not os.path.exists(person_path):
            return {"ok": False, "msg": "No existe la carpeta del estudiante"}
        
        # Asignar etiqueta si no existe
        if estudiante not in self.label_dict:
            self.label_dict[estudiante] = self.next_label
            self.image_paths.append(estudiante)
            self.next_label += 1
        
        label = self.label_dict[estudiante]
        
        # Cargar todas las fotos
        faces_data = []
        labels = []
        rutas_procesadas = []
        
        for filename in os.listdir(person_path):
            img_path = os.path.join(person_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces_data.append(img)
                labels.append(label)
                rutas_procesadas.append(img_path)
        
        if not faces_data:
            return {"ok": False, "msg": "No hay imágenes para entrenar"}
        
        # Entrenar incrementalmente
        try:
            self.face_recognizer.update(faces_data, np.array(labels))
            self.face_recognizer.write(str(self.model_path))
        except:
            # Si falla update (modelo vacío), usar train
            self.face_recognizer.train(faces_data, np.array(labels))
            self.face_recognizer.write(str(self.model_path))
        
        # Borrar imágenes temporales
        for ruta in rutas_procesadas:
            os.remove(ruta)
        
        # Intentar borrar carpeta si está vacía
        try:
            os.rmdir(person_path)
        except:
            pass
        
        print(f"Entrenamiento incremental: {len(faces_data)} imágenes de {estudiante}")
        
        return {
            "ok": True,
            "msg": "Modelo entrenado",
            "imagenes_procesadas": len(faces_data)
        }

    def entrenar_modelo_completo(self):
        """
        Re-entrena el modelo completo desde cero
        
        Returns:
            dict: Resultado del entrenamiento
        """
        print("Entrenando modelo completo...")
        
        # Listar personas en Data/
        people_list = [d for d in os.listdir(self.data_path) 
                      if os.path.isdir(os.path.join(self.data_path, d))]
        
        if not people_list:
            return {"ok": False, "msg": "No hay personas para entrenar"}
        
        labels = []
        faces_data = []
        label = 0
        
        for name_dir in people_list:
            person_path = os.path.join(self.data_path, name_dir)
            
            for filename in os.listdir(person_path):
                img_path = os.path.join(person_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces_data.append(img)
                    labels.append(label)
            
            label += 1
        
        if not faces_data:
            return {"ok": False, "msg": "No se encontraron imágenes"}
        
        # Entrenar
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_recognizer.train(faces_data, np.array(labels))
        self.image_paths = people_list
        self.label_dict = {name: idx for idx, name in enumerate(people_list)}
        self.next_label = len(people_list)
        
        # Guardar modelo
        self.face_recognizer.write(str(self.model_path))
        
        # Borrar imágenes originales
        for name_dir in people_list:
            person_path = os.path.join(self.data_path, name_dir)
            for filename in os.listdir(person_path):
                os.remove(os.path.join(person_path, filename))
            os.rmdir(person_path)
        
        print(f"Modelo entrenado con {len(faces_data)} imágenes")
        
        return {
            "ok": True,
            "msg": "Modelo entrenado completo",
            "personas": len(people_list),
            "imagenes_totales": len(faces_data)
        }

    def listar_estudiantes(self):
        """
        Retorna lista de estudiantes registrados en el modelo
        
        Returns:
            list: Lista de nombres de estudiantes
        """
        return self.image_paths.copy()

    def reset_reconocimientos(self):
        """Limpia el conjunto de estudiantes reconocidos (para nueva sesión)"""
        self.estudiantes_reconocidos.clear()
        self.tiempos_reconocimiento.clear()


# Singleton instance
reconocimiento_service = ReconocimientoService()