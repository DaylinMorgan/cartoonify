import cv2
import numpy as np

total_color = 5 #k-means
max_iter = 20
epsilon = 0.001

d = 50 # diameter of pixel neighborhood
sigmaColor = 200
sigmaSpace = 200

blur_value = 9
line_size = 201

def get_edges(img,line_size,blur_value):
    print('finding image edges')
    gray_blur = img
    #gray_blur = cv2.medianBlur(img, 21)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    #cv2.imwrite(f'edge_line{line_size}_blur{blur_value}.png',edges)
    cv2.imwrite('edges.png',edges)
    return edges

# effect is undesired color reduction
def color_quantization(img, k):
    print('transforming color space')
    # Transform the image
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-Means
    print('>>>implementing k-means')
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

def main():
    print('reading image')
    img = cv2.imread(r'headshot.png')

    img_newcolor = color_quantization(img, total_color)

    #blurred = cv2.bilateralFilter(img_newcolor, d=d, sigmaColor=sigmaColor,sigmaSpace=sigmaSpace)
    blurred = cv2.medianBlur(img_newcolor,11)

    edges = get_edges(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),line_size,blur_value)

    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    
    print('saving final image as cartoon.png')
    cv2.imwrite('cartoon.png',cartoon)

if __name__ == '__main__':
    main()