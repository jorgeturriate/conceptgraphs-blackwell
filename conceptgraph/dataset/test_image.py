import cv2
import numpy as np

# Carga una de tus imágenes de profundidad extraídas
depth = cv2.imread("/home/jorgeturriate/Docker_Images/noetic-stonefish-gtsam-v3/noetic-stonefish-gtsam/catkin_ws/bags/datasets/cirs_sequence/results/depth000000.png", cv2.IMREAD_UNCHANGED)

print(f"Tipo de datos: {depth.dtype}")  # Debería ser uint16
print(f"Distancia Mínima detectada: {np.min(depth[depth > 0])} mm") # Ej. ~1200 mm (1.2m)
print(f"Distancia Máxima detectada: {np.max(depth)} mm")            # Ej. ~4500 mm (4.5m)
print(f"Distancia Media: {np.mean(depth):.2f} mm")

# Para VISUALIZARLA como lo hace RViz/Replica (Normalización para ojo humano):
depth_visual = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite("depth_visual_test.png", depth_visual)
