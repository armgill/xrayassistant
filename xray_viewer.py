import cv2
import matplotlib.pyplot as plt


#load xray images from file
img = cv2.imread('file.jpg',cv2.IMREAD_GRAYSCALE)

# # check if image is loaded properly
if img is None:
    print("Oh no! Image failed to load. Check filename and path.")
    exit()

# display the image in figure window
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.show()


plt.figure(figsize=(12, 4))

# resize image
resized_img = cv2.resize(img,(1280,128))

plt.subplot(1, 3, 1)
plt.imshow(resized_img, cmap='gray')
plt.title("Resized")
plt.axis('off')

# noise reduction
blurred_img = cv2.GaussianBlur(resized_img, (5,5), 0)

plt.subplot(1, 3, 2)
plt.imshow(blurred_img, cmap='gray')
plt.title("Blurred")
plt.axis('off')

# edge detected
edges = cv2.Canny(blurred_img,50,150)

plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title("Edges")
plt.axis('off')

plt.tight_layout()
plt.show()
