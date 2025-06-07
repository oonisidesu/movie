#!/usr/bin/env python3
import cv2
import numpy as np

# Load the two results
triangulation_result = cv2.imread('demo_output/fixed_dlib_swap.jpg')
hybrid_result = cv2.imread('demo_output/improved_hybrid_swap.jpg')

# Resize for comparison if needed
if triangulation_result.shape != hybrid_result.shape:
    triangulation_result = cv2.resize(triangulation_result, (hybrid_result.shape[1], hybrid_result.shape[0]))

# Create side-by-side comparison
comparison = np.hstack([triangulation_result, hybrid_result])

# Add labels
cv2.putText(comparison, 'Triangulation (Kaleidoscope)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(comparison, 'Hybrid (Fixed)', (triangulation_result.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Save comparison
cv2.imwrite('demo_output/method_comparison.jpg', comparison)
print('Comparison saved to demo_output/method_comparison.jpg')