import matplotlib.pyplot as pyplot
import numpy


class Image:

    def __init__(self, size, title):
        self.size = size
        self.features = []
        self.images = []
        self.title = title
        self.background = None


    def set_instance(self, instance):
        self.images.append(numpy.zeros(self.size))
        self.images[-1] = numpy.reshape(instance, self.size)
        return self


    def set_background_instance(self, background):
        self.background = numpy.zeros(self.size)
        self.background = numpy.reshape(background, self.size)
        return self


    def add_reason(self, reason):
        self.images.append([numpy.zeros(self.size), numpy.zeros(self.size)])
        images = self.images[-1]
        with_weights = all(feature["weight"] is not None for feature in reason)
        if with_weights:
            max_weights = max(feature["weight"] for feature in reason if feature["weight"])
            min_weights = min(feature["weight"] for feature in reason if feature["weight"])

        for feature in reason:
            id_feature = feature["id"]
            sign = feature["sign"]
            weight = feature["weight"]

            x = (id_feature - 1) // self.size[0]
            y = (id_feature - 1) % self.size[1]
            color = (weight / (max_weights - min_weights)) * 256 if with_weights else 256
            if sign:
                images[0][x][y] = color
            else:
                images[1][x][y] = color

        images[0] = numpy.ma.masked_where(images[0] < 0.9, images[0])
        images[1] = numpy.ma.masked_where(images[1] < 0.9, images[1])
        return self


class Vizualisation():

    def __init__(self, x, y, instance=None):
        self.size = (x, y)
        self._heat_map_images = []


    '''
    Do not forget to convert an implicant in features thanks to tree.to_features(details=True).
    Create two images per reason, the positive one and the negative one.
    '''


    def new_image(self, title):
        self._heat_map_images.append(Image(self.size, title))
        return self._heat_map_images[-1]


    def display(self, *, n_rows=1):
        n_images = len(self._heat_map_images)
        if n_rows >= n_images:
            n_rows = 1
        fig, axes = pyplot.subplots(n_rows,
                                    n_images if n_rows == 1 else n_images // n_rows + (1 if n_images % n_rows != 0 else 0),
                                    figsize=self.size)
        for i, heat_map_images in enumerate(self._heat_map_images):
            if n_images == 1:
                axes.title.set_text(heat_map_images.title)
            else:
                axes.flat[i].title.set_text(heat_map_images.title)
            for image in heat_map_images.images:
                if isinstance(image, list):
                    if n_images == 1:
                        axes.imshow(image[0], alpha=0.6, cmap='Blues', vmin=0, vmax=255, interpolation='None')
                        axes.imshow(image[0], alpha=0.6, cmap='Reds', vmin=0, vmax=255, interpolation='None')
                    else:
                        a = axes if n_rows == 1 else axes[i // n_rows]
                        idx = i if n_rows == 1 else i % n_rows
                        a.flat[idx].imshow(image[0], alpha=0.6, cmap='Blues', vmin=0, vmax=255, interpolation='None')
                        a.flat[idx].imshow(image[1], alpha=0.6, cmap='Reds', vmin=0, vmax=255, interpolation='None')
                        if heat_map_images.background is not None:
                            a.flat[idx].imshow(heat_map_images.background, alpha=0.2, cmap='Greys', vmin=0, vmax=255)
                else:
                    if n_images == 1:
                        axes.imshow(image)
                    else:
                        axes.flat[i].imshow(image)
        pyplot.show()


    #def display_observation(self):
        #        pyplot.imshow(self.image_observation)
    #    pyplot.show()
