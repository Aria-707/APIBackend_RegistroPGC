"""
Vistas de la API de reconocimiento facial
"""
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .services.reconocimiento_service import reconocimiento_service
from .services.asistencia_service import asistencia_service
from .permissions import verificar_token
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name='dispatch')
class RegistroView(APIView):
    """
    Endpoint para reconocimiento facial en tiempo real
    
    POST: Recibe imagen en base64 y retorna si reconoce a alguien
    """
    
    def post(self, request):
        # Verificar token
        token_error = verificar_token(request)
        if token_error:
            return token_error
        
        try:
            image_data = request.data.get('image')
            
            if not image_data:
                return Response(
                    {"estado": "error", "mensaje": "No se recibió imagen"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            resultado = reconocimiento_service.reconocer_rostro(image_data)
            return Response(resultado, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {"estado": "error", "mensaje": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class GuardarFotoView(APIView):
    """
    Endpoint para guardar fotos durante el registro de un estudiante
    
    POST: Recibe nombre de estudiante y foto en base64
    """
    
    def post(self, request):
        # Verificar token
        token_error = verificar_token(request)
        if token_error:
            return token_error
        
        try:
            estudiante = request.data.get('estudiante', '').strip()
            foto = request.data.get('foto', '')
            
            if not estudiante or not foto:
                return Response(
                    {"error": "Faltan datos"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            resultado = reconocimiento_service.guardar_foto_registro(estudiante, foto)
            
            if resultado['ok']:
                return Response(resultado, status=status.HTTP_200_OK)
            else:
                return Response(resultado, status=status.HTTP_200_OK)
                
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class HealthCheckView(APIView):
    """
    GET /api/health/
    Verifica que el servidor esté funcionando
    """
    
    def get(self, request):
        logger.info("💚 [HEALTH CHECK] Verificando estado del servidor")
        
        # NO verificar token en health check
        
        try:
            # Verificar conexión a Firebase
            from ..firebase_config import db
            docs_count = len(list(db.collection("asistenciaReconocimiento").limit(1).stream()))
            firebase_status = "✅ Conectado"
        except Exception as e:
            firebase_status = f"❌ Error: {str(e)}"
            logger.error(f"❌ Firebase error: {str(e)}")
        
        return Response({
            "status": "OK",
            "timestamp": datetime.now().isoformat(),
            "firebase": firebase_status,
            "endpoints": {
                "reconocimiento": "/api/registro/",
                "guardar_foto": "/api/guardar-foto/",
                "entrenar": "/api/entrenar/",
                "asistencias": "/api/asistencia/consultar/",
                "registrar_asistencia": "/api/asistencia/registrar/",
                "estudiantes": "/api/estudiantes/"
            }
        }, status=status.HTTP_200_OK)


@method_decorator(csrf_exempt, name='dispatch')
class EntrenarModeloView(APIView):
    """
    Endpoint para entrenar el modelo después de capturar fotos
    
    POST: Recibe nombre de estudiante y entrena el modelo
    """
    
    def post(self, request):
        # Verificar token
        token_error = verificar_token(request)
        if token_error:
            return token_error
        
        try:
            estudiante = request.data.get('estudiante', '').strip()
            tipo_entrenamiento = request.data.get('tipo', 'incremental')  # 'incremental' o 'completo'
            
            if not estudiante and tipo_entrenamiento == 'incremental':
                return Response(
                    {"error": "Se requiere nombre de estudiante para entrenamiento incremental"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if tipo_entrenamiento == 'completo':
                resultado = reconocimiento_service.entrenar_modelo_completo()
            else:
                resultado = reconocimiento_service.entrenar_incremental(estudiante)
            
            if resultado['ok']:
                return Response(resultado, status=status.HTTP_200_OK)
            else:
                return Response(resultado, status=status.HTTP_400_BAD_REQUEST)
                
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class RegistrarAsistenciaView(APIView):
    """
    Endpoint para registrar asistencia manualmente
    
    POST: Registra asistencia de un estudiante
    """
    
    def post(self, request):
        # Verificar token
        token_error = verificar_token(request)
        if token_error:
            return token_error
        
        try:
            estudiante = request.data.get('estudiante')
            estado_asistencia = request.data.get('estadoAsistencia', 'Presente')
            asignatura = request.data.get('asignatura', 'Física')
            
            if not estudiante:
                return Response(
                    {"error": "Se requiere nombre del estudiante"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            resultado = asistencia_service.registrar_asistencia(
                estudiante,
                estado_asistencia,
                asignatura
            )
            
            return Response(resultado, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class ConsultarAsistenciaView(APIView):
    """
    Endpoint para consultar asistencias
    
    GET: Retorna lista de asistencias con filtros opcionales
    """
    
    def get(self, request):
        # Verificar token
        token_error = verificar_token(request)
        if token_error:
            return token_error
        
        try:
            # Obtener parámetros de consulta
            estudiante = request.query_params.get('estudiante', None)
            asistencia_id = request.query_params.get('id', None)
            
            # Si se proporciona ID, consultar asistencia específica
            if asistencia_id:
                resultado = asistencia_service.obtener(asistencia_id)
                if resultado:
                    return Response(resultado, status=status.HTTP_200_OK)
                else:
                    return Response(
                        {"error": "Asistencia no encontrada"},
                        status=status.HTTP_404_NOT_FOUND
                    )
            
            # Consultar con filtros
            if estudiante:
                asistencias = asistencia_service.consultar_por_estudiante(estudiante)
            else:
                asistencias = asistencia_service.listar_todas()
            
            return Response(
                {"asistencias": asistencias, "total": len(asistencias)},
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class ListarEstudiantesView(APIView):
    """
    Endpoint para listar estudiantes registrados en el modelo
    
    GET: Retorna lista de estudiantes
    """
    
    def get(self, request):
        # Verificar token
        token_error = verificar_token(request)
        if token_error:
            return token_error
        
        try:
            estudiantes = reconocimiento_service.listar_estudiantes()
            
            return Response(
                {
                    "estudiantes": estudiantes,
                    "total": len(estudiantes)
                },
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )