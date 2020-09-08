from PIL import Image


def blend_two_images2():
    img1 = Image.open("/Volumes/Other/paper3/multi-task.chainer/datasets/demo/1.jpg")
    img1 = img1.convert('RGBA')

    img2 = Image.open("/Volumes/Other/paper3/multi-task.chainer/datasets/demo/000002.jpg")
    img2 = img2.convert('RGBA')

    r, g, b, alpha = img2.split()
    alpha = alpha.point(lambda i: i > 0 and 204)

    img = Image.composite(img2, img1, alpha)

    img.show()
    #img.save("blend2.png")

    return


if __name__ == '__main__':
    blend_two_images2()
