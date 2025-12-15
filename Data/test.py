import cv2
import matplotlib.pyplot as plt
import numpy as np
from inference_sdk import InferenceHTTPClient

image_path = "/Users/ofekhazum/Desktop/workArea/07_12_25/documents/2ejatYsIdLtbMg8_doc_front.jpg"
CONFIDENCE_THRESHOLD = 0.7

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="v5nYPJVr4VP4XoJRmdUs"
)

result = client.run_workflow(
    workspace_name="ohfootballanalytics",
    workflow_id="find-cards",
    images={
        "image": image_path
    },
    use_cache=True # cache workflow definition for 15 minutes
)

predictions = result[0]['output']['predictions']

# for prediction in predictions:
    
COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), 
    (0, 255, 255), (255, 255, 0), (255, 0, 255)
]

# 2. LOAD IMAGE
# image = cv2.imread(image_path)

# if image is None:
#     print("Error: Image not found. Check the path.")
# else:
#     # 3. GENERIC LOOP
#     for i, pred in enumerate(predictions):
        
#         # Extract confidence
#         conf = pred.get('confidence', 0.0)
        
#         # --- FILTER: CHECK CONFIDENCE ---
#         if conf < CONFIDENCE_THRESHOLD:
#             continue  # Skip this item if confidence is too low
            
#         # Select color based on index
#         color = COLORS[i % len(COLORS)]
        
#         # Extract data
#         label = pred.get('class', str(pred.get('class_id', 'Object')))
        
#         # Coordinates (Center X, Y)
#         c_x = pred['x']
#         c_y = pred['y']
#         w = pred['width']
#         h = pred['height']

#         # Calculate Corners
#         x1 = int(c_x - (w / 2))
#         y1 = int(c_y - (h / 2))
#         x2 = int(c_x + (w / 2))
#         y2 = int(c_y + (h / 2))

#         # 4. DRAWING
#         cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=4)

#         text = f"{label}: {conf:.2f}"
#         (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
#         cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
#         cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
#                     1.5, (255, 255, 255), 3)

#     # 5. DISPLAY
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     plt.figure(figsize=(12, 12))
#     plt.imshow(image_rgb)
#     plt.axis('off')
#     plt.show()

# -----------------------------BINARY MASK GENERATION-----------------------------

# image = cv2.imread(image_path)

# if image is None:
#     print("Error: Image not found.")
# else:
#     # Get image dimensions (Height, Width)
#     h, w = image.shape[:2]

#     # 3. CREATE BLANK MASK
#     # Create a black image (zeros) with the same size as the original
#     # dtype=np.uint8 is standard for images (0-255 range)
#     mask = np.zeros((h, w), dtype=np.uint8)

#     for pred in predictions:
#         # Filter by confidence
#         if pred.get('confidence', 0) < CONFIDENCE_THRESHOLD:
#             continue

#         # Check if 'points' data exists (Instance Segmentation)
#         if 'points' in pred:
#             # Extract points into a list of [x, y]
#             pts_list = [[p['x'], p['y']] for p in pred['points']]
            
#             # Convert to numpy array of integers (required by cv2.fillPoly)
#             pts_np = np.array(pts_list, dtype=np.int32)
            
#             # Reshape is sometimes needed for fillPoly (n, 1, 2), 
#             # though often works without it depending on cv2 version.
#             pts_np = pts_np.reshape((-1, 1, 2))

#             # Draw the filled polygon onto the mask
#             # 255 means "White"
#             cv2.fillPoly(mask, [pts_np], 255)
            
#         else:
#             # FALLBACK: If 'points' is missing, use the bounding box
#             c_x, c_y = pred['x'], pred['y']
#             bw, bh = pred['width'], pred['height']
#             x1 = int(c_x - (bw / 2))
#             y1 = int(c_y - (bh / 2))
#             x2 = int(c_x + (bw / 2))
#             y2 = int(c_y + (bh / 2))
            
#             # Draw filled rectangle (-1 means fill)
#             cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

#     # 4. (OPTIONAL) EXTRACT THE CARD
#     # Use the mask to cut the card out of the original image
#     # 'bitwise_and' keeps pixels where mask is white, blacks out the rest
#     masked_image = cv2.bitwise_and(image, image, mask=mask)

#     # 5. DISPLAY RESULTS
#     # Convert BGR to RGB for Matplotlib
#     masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    
#     # Show Mask and Result side-by-side
#     plt.figure(figsize=(15, 10))
    
#     plt.subplot(1, 2, 1)
#     plt.title("Generated Binary Mask")
#     plt.imshow(mask, cmap='gray') # Display in grayscale
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.title("Mask Applied to Image")
#     plt.imshow(masked_image_rgb)
#     plt.axis('off')
    
#     plt.show()


# -----------------------------BLUR GENERATION-----------------------------

BLUR_STRENGTH = (301, 301)

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
else:
    h, w = image.shape[:2]

    # --- 3. GENERATE THE MASK (Same logic as before) ---
    mask = np.zeros((h, w), dtype=np.uint8)

    for pred in predictions:
        if pred.get('confidence', 0) < CONFIDENCE_THRESHOLD:
            continue

        if 'points' in pred:
            pts_list = [[p['x'], p['y']] for p in pred['points']]
            pts_np = np.array(pts_list, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts_np], 255)
        else:
            # Fallback for boxes if points are missing
            c_x, c_y = pred['x'], pred['y']
            bw, bh = pred['width'], pred['height']
            x1, y1 = int(c_x - bw/2), int(c_y - bh/2)
            x2, y2 = int(c_x + bw/2), int(c_y + bh/2)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)


    # --- 4. CREATE BLURRED IMAGE & COMBINE ---

    # A. Create a heavily blurred version of the ENTIRE image
    blurred_source = cv2.GaussianBlur(image, BLUR_STRENGTH, 100)

    # B. Prepare the mask for combining
    # The image has 3 channels (BGR), but the mask only has 1 (Grayscale).
    # We need to make the mask 3 channels so their shapes match for the math.
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # C. The Combination Logic (numpy.where)
    # This is an element-wise if/else statement:
    # "Where mask_3ch is greater than 0 (white), use the blurred_source pixel.
    #  Otherwise (black), use the original image pixel."
    final_result = np.where(mask_3ch > 0, blurred_source, image)


    # --- 5. DISPLAY ---
    # Convert BGR to RGB for Matplotlib display
    final_result_rgb = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 2, 1)
    plt.title("The Mask Used")
    plt.imshow(mask_display)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Final Blurred Result")
    plt.imshow(final_result_rgb)
    plt.axis('off')
    
    plt.show()