"""
Verificación de tokens de autenticación con Firebase
"""
from firebase_admin import auth
from rest_framework.response import Response
from rest_framework import status
import logging

logger = logging.getLogger(__name__)


def verificar_token(request):
    """
    Verifica el token de autenticación de Firebase
    """
    auth_header = request.headers.get('Authorization')
    
    if not auth_header:
        logger.warning("Petición sin token de autorización")
        return Response(
            {"error": "No se encontró el token de autorización."},
            status=status.HTTP_401_UNAUTHORIZED
        )
    
    parts = auth_header.split(' ')
    if len(parts) != 2 or parts[0] != 'Bearer':
        logger.warning("Formato de token inválido")
        return Response(
            {"error": "Formato de token inválido. Usa 'Bearer <token>'"},
            status=status.HTTP_401_UNAUTHORIZED
        )
    
    id_token = parts[1]
    
    try:
        decoded_token = auth.verify_id_token(id_token)
        request.user_firebase = decoded_token
        logger.info(f"Token verificado para usuario: {decoded_token.get('email', 'N/A')}")
        return None
        
    except auth.InvalidIdTokenError:
        logger.error("Token inválido")
        return Response(
            {"error": "Token inválido."},
            status=status.HTTP_401_UNAUTHORIZED
        )
    except auth.ExpiredIdTokenError:
        logger.error("Token expirado")
        return Response(
            {"error": "Token expirado."},
            status=status.HTTP_401_UNAUTHORIZED
        )
    except auth.RevokedIdTokenError:
        logger.error("Token revocado")
        return Response(
            {"error": "Token revocado."},
            status=status.HTTP_401_UNAUTHORIZED
        )
    except Exception as e:
        logger.error(f"Error verificando token: {str(e)}")
        return Response(
            {"error": f"Error verificando token: {str(e)}"},
            status=status.HTTP_401_UNAUTHORIZED
        )