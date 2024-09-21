from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os

# Cargar el modelo previamente entrenado
model = load_model('tasks/resultados/distemper_dense_model.h5')
# Definir el número de características
n_features = 12

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            # Obtener los datos de la solicitud
            data = json.loads(request.body)

            # Verificar si los datos tienen la estructura correcta
            if 'features' not in data:
                return JsonResponse({'error': 'Datos incompletos o incorrectos'}, status=400)

            # Verificar que las características tengan el tamaño adecuado (n_features)
            if len(data['features']) != n_features:
                return JsonResponse({'error': f'Datos incorrectos, se esperaban {n_features} características, pero se recibieron {len(data["features"])}'}, status=400)

            # Convertir las características a un arreglo de numpy
            features = np.array(data['features']).reshape(1, n_features)

            # Hacer la predicción usando el modelo cargado
            prediction = model.predict(features)

            # Convertir la predicción a 1 (positivo) o 0 (negativo) según el umbral (0.7)
            pred_class = int(prediction[0][0] >= 0.7)
            confidence = float(prediction[0][0])  # Convertir a float para evitar errores de serialización

            # Devolver los resultados como JSON
            return JsonResponse({
                'prediction': pred_class,
                'confidence': confidence
            })

        except Exception as e:
            # En caso de error, devolver el mensaje de error
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Método no permitido'}, status=405)