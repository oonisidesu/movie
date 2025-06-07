import cv2
import numpy as np

# Load comparison images
center = cv2.imread('demo_output/center_adjusted_swap.jpg')
left_adjusted = cv2.imread('demo_output/left_adjusted_swap.jpg')
natural = cv2.imread('demo_output/natural_hybrid_swap.jpg')

# Create three-way comparison
comparison = np.hstack([center, natural, left_adjusted])

# Add labels
cv2.putText(comparison, 'Center (0.0)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.putText(comparison, 'Default (0.15)', (center.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(comparison, 'Left More (0.25)', (center.shape[1] + natural.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Save comparison
cv2.imwrite('demo_output/position_comparison.jpg', comparison)
print('Position adjustment comparison saved')