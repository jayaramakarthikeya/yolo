import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def plot_bboxes(images,labels_):
    # labels: [8, 8, 8]
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    classes = ['Pedestrian', 'Traffic Light', 'Car']
    plt.figure(figsize=(10,10))
    n = len(images)
    for i in range(len(images)):
        image = images[i].copy()
        label = labels_[i].astype(int).tolist()
        for j in range(len(label)):
            cv2.rectangle(image, (label[j][1], label[j][2]), (label[j][3], label[j][4]), colors[label[j][0]], 1)
        
        
        # Plot the image
        plt.subplot(1, n, i+1)
        plt.imshow(image)
        plt.axis('off')  # Turn off grid numbering

        # Create legend
        red_patch = mpatches.Patch(color='red', label='Pedestrian')
        green_patch = mpatches.Patch(color='green', label='Traffic Light')
        blue_patch = mpatches.Patch(color='blue', label='Car')
        
        plt.legend(handles=[red_patch, green_patch, blue_patch], loc='upper right')
    plt.show()