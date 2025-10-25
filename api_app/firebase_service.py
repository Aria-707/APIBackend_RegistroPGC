import firebase_admin
from firebase_admin import credentials, firestore
from django.conf import settings

# Inicializar Firebase solo una vez
if not firebase_admin._apps:
    cred = credentials.Certificate(
        settings.FIREBASE_CREDENTIALS_PATH
    )
    firebase_admin.initialize_app(cred)

# Cliente de Firestore
db = firestore.client()