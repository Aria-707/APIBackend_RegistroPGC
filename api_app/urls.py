"""
URLs de la API de reconocimiento facial
"""
from django.urls import path
from . import views

app_name = 'api'

urlpatterns = [
    # Health check (sin autenticación)
    path('health/', views.HealthCheckView.as_view(), name='health_check'),
    
    # Endpoints de reconocimiento facial
    path('registro/', views.RegistroView.as_view(), name='registro'),
    path('guardar-foto/', views.GuardarFotoView.as_view(), name='guardar_foto'),
    path('entrenar/', views.EntrenarModeloView.as_view(), name='entrenar'),
    
    # Endpoints de asistencias
    path('asistencia/registrar/', views.RegistrarAsistenciaView.as_view(), name='registrar_asistencia'),
    path('asistencia/consultar/', views.ConsultarAsistenciaView.as_view(), name='consultar_asistencia'),
]