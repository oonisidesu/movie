import cv2
import numpy as np

# Load comparison images
previous = cv2.imread('demo_output/enhanced_hybrid_swap.jpg')
improved = cv2.imread('demo_output/natural_hybrid_swap.jpg')

# Resize for comparison
if previous.shape != improved.shape:
    previous = cv2.resize(previous, (improved.shape[1], improved.shape[0]))

# Create side-by-side comparison
comparison = np.hstack([previous, improved])

# Add labels
cv2.putText(comparison, 'Previous', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
cv2.putText(comparison, 'Natural Improved', (previous.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Save comparison
cv2.imwrite('demo_output/natural_comparison.jpg', comparison)
print('Natural improvement comparison saved')