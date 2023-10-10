import matplotlib.pyplot as pyplot
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import numpy
from PIL import Image as PILImage
from PIL.ImageQt import ImageQt
import os
import copy

class PyPlotDiagramGenerator():
    def __init__(self, time_series=None):
        self.time_series=time_series

    def convert_features_to_dict_features(self, features):
        dict_features = dict()
        for feature in features:
            name = feature["name"]
            if name not in dict_features.keys():
                dict_features[name] = [feature]
            else:
                dict_features[name].append(feature)
        return dict_features
    
    def generate_explanation(self, feature_values, instance, reason):

        color_blue = "#4169E1"
        color_red = "#CD5C5C"
        color_grey = "#808080"
        trans_left = mpl.transforms.Affine2D().translate(-5, 0)
        trans_right = mpl.transforms.Affine2D().translate(5, 0)
        
        mpl.rcParams['axes.spines.left'] = True
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.bottom'] = True
        dict_features = copy.deepcopy(reason)
        
        print("self.time_series:", self.time_series)
        n_subplots_time_series = 0
        _time_series = copy.deepcopy(self.time_series)
        if _time_series is not None:
            #remove of dict_features all features of the time series
            n_subplots_time_series+=len(_time_series.keys())
            
            for key in _time_series.keys():
                for i, feature in enumerate(_time_series[key]):
                    if feature not in dict_features.keys():
                        _time_series[key][i] = feature
                        continue
                    dict_features_value = dict_features[feature]  
                    _time_series[key][i] = dict_features_value.copy()                    
                    dict_features.pop(feature)

        n_subplots = len(dict_features.keys()) + n_subplots_time_series
        ratio_subplots = [4]*n_subplots_time_series+[1]*len(dict_features.keys())
        if len(dict_features.keys()) == 1:
            fig, axes = pyplot.subplots(n_subplots, figsize=(6,1))
            axes = [axes] # To solve a bug when axes is not a list. 
        else:
            fig, axes = pyplot.subplots(n_subplots, figsize=(6,n_subplots+3), gridspec_kw={'height_ratios': ratio_subplots})
        print(feature_values)
        if _time_series is not None:
            # time_series graphs
            
            for id_axe, key in enumerate(_time_series.keys()):
                title = key
                instance_values = []
                feature_names = []
                explanation_min_values = []
                explanation_max_values = []
                
                for i, info in enumerate(_time_series[key]):
                    if isinstance(info, str):
                        feature_name = info
                    else:
                        feature_name = info[0]["name"]
                    
                    instance_value = feature_values[feature_name]
                    instance_values.append(instance_value)
                    feature_names.append(feature_name)
                    if isinstance(info, str):
                        explanation_min_values.append("inf")
                        explanation_max_values.append("inf")
                        continue
                    string_view = info[0]["string"]
                    theory = info[0]["theory"] 

                    if theory is not None and (theory[0] == "binary" or theory[0] == "categorical"):
                        raise ValueError("The feature of time series must be None or Numerical.")
                    
                    print("string_view:", string_view)
                    print("feature_name:", feature_name)
                    if "in [" in string_view or "in ]" in string_view:
                        feature_str, interval = string_view.split("in")
                        feature_str = feature_str.lstrip().rstrip()
                        
                        threshold_1, threshold_2 = interval.split(", ")
                        threshold_str_1 = threshold_1
                        threshold_str_2 = threshold_2
                        threshold_1 = float(threshold_1.replace("[", "").replace("]", "").replace(",", ""))
                        threshold_2 = float(threshold_2.replace("[", "").replace("]", "").replace(",", ""))
                        bound_left = min(threshold_1, threshold_2, instance_value)
                        bound_right = max(threshold_1, threshold_2, instance_value)
                        total = bound_right - bound_left
                        bound_left = bound_left - (total/10)
                        bound_right = bound_right + (total/10)
                        explanation_min_values.append(threshold_1)
                        explanation_max_values.append(threshold_2)
                        
                    else:
                        #Case with a simple condition feature > threshold
                        
                        for operator in ["<=", ">=", "<", ">", "==", "!=", "="]:
                            if operator in string_view:
                                feature_str, threshold_str = string_view.split(operator)
                                feature_str = feature_str.lstrip().rstrip()
                                operator_str = operator
                                threshold = float(threshold_str)
                                break  
                        if operator == "<=" or operator == "<":
                            threshold_max = threshold
                            threshold_min = "inf"
                        elif operator == ">=" or operator == ">":
                            threshold_max = "inf"
                            threshold_min = threshold
                        explanation_min_values.append(threshold_min)
                        explanation_max_values.append(threshold_max)
                    
                    
                print("instance_values:", instance_values)

                midle = (0+len(instance_values)-1)/2
                print("here:", midle)
                to_min = [x for x in instance_values if x != "inf"] + [x for x in explanation_min_values if x != "inf"] + [x for x in explanation_max_values if x != "inf"] 
                to_max = [x for x in instance_values if x != "inf"] + [x for x in explanation_min_values if x != "inf"] + [x for x in explanation_max_values if x != "inf"]
                min_y = numpy.min(to_min)
                max_y = numpy.max(to_max)
                
                margin_y = (numpy.abs(max_y - min_y))/10
                
                explanation_min_values = [min_y-margin_y if x == "inf" else x for x in explanation_min_values]
                explanation_max_values = [max_y+margin_y if x == "inf" else x for x in explanation_max_values]
                
                axes[id_axe].set_ylim(bottom=min_y-margin_y, top=max_y+margin_y)
                axes[id_axe].set_xlim(left=0, right=len(instance_values)-1)
                axes[id_axe].plot(feature_names,instance_values, color=color_red)
                axes[id_axe].plot(feature_names,explanation_min_values, color=color_blue)
                axes[id_axe].plot(feature_names,explanation_max_values, color=color_blue)
                axes[id_axe].set_title(title, fontsize=10)
                #axes[id_axe].text(midle,numpy.min(instance_values)-margin_y-margin_y,title)
        
        for i, feature in enumerate(dict_features.keys()):
            i = i + n_subplots_time_series
            if "string" not in dict_features[feature][0].keys():
                raise ValueError("The string version of this feature is not done: " + feature)
            string_view = dict_features[feature][0]["string"]
            theory = dict_features[feature][0]["theory"] 
            
            bound_left = 0
            bound_right = 1
            bound_explanation_left = None
            bound_explanation_right = None
            if theory is not None:
                if theory[0] == "binary":
                    #binary case
                    feature_str, operator_str, threshold_str = string_view.split(" ")
                    
                    if operator_str == "=":
                        txt = "False" if threshold_str == "0" else "True"
                        #colors = [color_blue, color_red] if threshold_str == "0" else [color_red, color_blue]
                    else:
                        txt = "True" if threshold_str == "0" else "False"
                        #colors = [color_red, color_blue] if threshold_str == "0" else [color_blue, color_red]
                    txt = feature_str + " is " + txt 
                    the_table = axes[i].table(
                        cellText=[[txt]],
                        #cellColours=[colors],
                        loc='center',
                        colLoc='center',
                        cellLoc='center',
                        bbox=[0, -0.3, 1, 0.275])
                    
                    
                    the_table.set_fontsize(10)
                    #axes[i].text(0.5,0.2,feature_str)
                    
                    axes[i].yaxis.set_visible(False)
                    axes[i].xaxis.set_visible(False)
                    axes[i].axis('off')
                    continue
                elif theory[0] == "categorical":
                    #categorical case
                    feature_str, operator_str, values = string_view.split(" ")
                    all_values = [str(element) for element in theory[1][2]]
                    colors = [color_grey] * len(all_values)
                    if operator_str == "=":
                        index = all_values.index(values)
                        for c in range(len(colors)): colors[c] = color_red
                        colors[index] = color_blue
                    elif operator_str == "!=":
                        if "{" in values:
                            values_to_red = [element for element in values.replace("{","").replace("}","").split(",")]
                            for value in values_to_red:
                                index = all_values.index(value)
                                colors[index] = color_red
                        else:
                            index = all_values.index(values)
                            colors[index] = color_red

                    the_table = axes[i].table(
                        cellText=[all_values],
                        cellColours=[colors],
                        loc='center',
                        colLoc='center',
                        cellLoc='center',
                        bbox=[0, -0.3, 1, 0.275])
                    the_table.set_fontsize(10)
                    axes[i].text(0.5,0.2,feature_str)
                    
                    axes[i].yaxis.set_visible(False)
                    axes[i].xaxis.set_visible(False)
                    axes[i].axis('off')
                    
                    continue

            value_instance = feature_values[feature]
            #numerical case: if theory is None, all features are considered as numerical
            if "in [" in string_view or "in ]" in string_view:
                if "and" in string_view:
                    #case feature in [infinty, threshold1] and feature in [threshold2, infinity]
                    raise NotImplementedError("TODO feature in [infinty, threshold1] and feature in [threshold2, infinity]")
                else:
                    #case feature in [threshold1, threshold2]
                    
                    feature_str, interval = string_view.split("in")
                    feature_str = feature_str.lstrip().rstrip()
                    
                    threshold_1, threshold_2 = interval.split(", ")
                    threshold_str_1 = threshold_1
                    threshold_str_2 = threshold_2
                    threshold_1 = float(threshold_1.replace("[", "").replace("]", "").replace(",", ""))
                    threshold_2 = float(threshold_2.replace("[", "").replace("]", "").replace(",", ""))

                    bound_left = min(threshold_1, threshold_2, value_instance)
                    bound_right = max(threshold_1, threshold_2, value_instance)
                    total = bound_right - bound_left
                    bound_left = bound_left - (total/10)
                    bound_right = bound_right + (total/10)

                    midle = (bound_left+bound_right)/2
                    axes[i].set_ylim(bottom=-1, top=50)
                    axes[i].set_xlim(left=bound_left, right=bound_right)
                    axes[i].plot([bound_left, bound_right], [25, 25], color=color_blue)
                    axes[i].text(midle,35,feature)
                    axes[i].yaxis.set_visible(False)
                    
                    bracket_left = None
                    bound_explanation_left = min(threshold_1, threshold_2)
                    if bound_explanation_left == threshold_1:
                        bracket_left = "$\mathcal{[}$" if "[" in threshold_str_1 else "$\mathcal{]}$"
                    else:
                        bracket_left = "$\mathcal{[}$" if "[" in threshold_str_2 else "$\mathcal{]}$"

                    bracket_right = None
                    bound_explanation_right = max(threshold_1, threshold_2)
                    if bound_explanation_right == threshold_1:
                        bracket_right = "$\mathcal{[}$" if "[" in threshold_str_1 else "$\mathcal{]}$"
                    else:
                        bracket_right = "$\mathcal{[}$" if "[" in threshold_str_2 else "$\mathcal{]}$"

                    axes[i].plot([bound_explanation_left, bound_explanation_right], [25, 25], color=color_blue, linewidth=5)

                    axes[i].plot(value_instance, 25, marker="o", color=color_red, clip_on=False, markersize=10)
                    
                    if bracket_left is not None:
                        x = axes[i].plot(bound_explanation_left, 25, marker=bracket_left, color=color_blue, clip_on=False, markersize=20)
                        if bracket_left == "$\mathcal{]}$": x[0].set_transform(x[0].get_transform()+trans_left)
                    if bracket_right is not None:
                        axes[i].plot(bound_explanation_right, 25, marker=bracket_right, color=color_blue, clip_on=False, markersize=20)
                        if bracket_right == "$\mathcal{[}$": x[0].set_transform(x[0].get_transform()+trans_right)
                    
            else:
                #Case with a simple condition feature > threshold
                
                for operator in ["<=", ">=", "<", ">", "==", "!=", "="]:
                    if operator in string_view:
                        feature_str, threshold_str = string_view.split(operator)
                        feature_str = feature_str.lstrip().rstrip()
                        operator_str = operator
                        break
                
                if operator_str in ["!=", "==", "="]:
                    if operator_str == "=" or operator_str == "==":
                        txt = "False" if threshold_str == "0" else "True"
                        #colors = [color_blue, color_red] if threshold_str == "0" else [color_red, color_blue]
                    else:
                        txt = "True" if threshold_str == "0" else "False"
                        #colors = [color_red, color_blue] if threshold_str == "0" else [color_blue, color_red]
                    txt = feature_str + " is " + txt 
                    the_table = axes[i].table(
                        cellText=[[txt]],
                        #cellColours=[colors],
                        loc='center',
                        colLoc='center',
                        cellLoc='center',
                        bbox=[0, -0.3, 1, 0.275])
                    
                    
                    the_table.set_fontsize(10)
                    #axes[i].text(0.5,0.2,feature_str)
                    
                    axes[i].yaxis.set_visible(False)
                    axes[i].xaxis.set_visible(False)
                    axes[i].axis('off')
                    continue
                threshold = float(threshold_str)
                bound_left = min(threshold, value_instance)
                bound_right = max(threshold, value_instance)
                total = bound_right - bound_left
                if bound_right == bound_left:
                    total = bound_right
                bound_left = bound_left - (total/10)
                bound_right = bound_right + (total/10)
                
                
                midle = (bound_left+bound_right)/2
                axes[i].set_ylim(bottom=-1, top=50)
                axes[i].set_xlim(left=bound_left, right=bound_right)
                axes[i].plot([bound_left, bound_right], [25, 25], color=color_blue)
                axes[i].text(midle,35,feature)
                axes[i].yaxis.set_visible(False)
                
                bracket_left = None
                bracket_right = None
                if operator_str == ">" or operator_str == ">=":
                    bound_explanation_left = threshold
                    bound_explanation_right = bound_right
                    bracket_left = "$\mathcal{[}$" if operator_str == ">=" else "$\mathcal{]}$"
                
                elif operator_str == "<" or operator_str == "<=":
                    bound_explanation_left = bound_left
                    bound_explanation_right = threshold
                    bracket_right = "$\mathcal{[}$" if operator_str == "<" else "$\mathcal{]}$"
                
                else:
                    raise NotImplementedError("This operator is not take into account: "+ operator_str)
                
                axes[i].plot([bound_explanation_left, bound_explanation_right], [25, 25], color=color_blue, linewidth=5)
                axes[i].plot(value_instance, 25, marker="o", color=color_red, clip_on=False, markersize=10)
                
                if bracket_left is not None:
                    x = axes[i].plot(threshold, 25, marker=bracket_left, color=color_blue, clip_on=False, markersize=20)
                    if bracket_left == "$\mathcal{]}$": x[0].set_transform(x[0].get_transform()+trans_left)
                
                if bracket_right is not None:
                    x = axes[i].plot(threshold, 25, marker=bracket_right, color=color_blue, clip_on=False, markersize=20)
                    if bracket_right == "$\mathcal{[}$": x[0].set_transform(x[0].get_transform()+trans_right)
                        
                        

        pyplot.subplots_adjust(top=1, hspace=1)

        pyplot.savefig('tmp_diagram.png', bbox_inches='tight')
        image = PILImage.open('tmp_diagram.png')
        pyplot.close()
        os.remove('tmp_diagram.png')
        
        return ImageQt(image)

class PyPlotImageGenerator():

    def __init__(self, image):
        self.image = image
        self.size = (self.image["shape"][0], self.image["shape"][1])
        self.n_colors = numpy.iinfo(self.image["dtype"]).max
        self.get_pixel_value = self.image["get_pixel_value"]
        self.instance_index_to_pixel_position = self.image["instance_index_to_pixel_position"]
    
    def instance_to_numpy(self, instance):
        image = numpy.zeros(self.image["shape"], dtype=self.image["dtype"])
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                image[i,j] = self.get_pixel_value(instance, i, j, self.image["shape"])
        return image 
    
    def generate_instance(self, instance):
        array_image = self.instance_to_numpy(instance)
        self.PIL_instance = PILImage.fromarray(array_image)
        return ImageQt(self.PIL_instance)
    
    def generate_explanation(self, instance, reason):
        reason = [reason[key][0] for key in reason.keys()]
        self.image_positive = numpy.zeros(self.size)
        self.image_negative = numpy.zeros(self.size)
        with_weights = all(feature["weight"] is not None for feature in reason)
        if with_weights:
            max_weights = max(feature["weight"] for feature in reason if feature["weight"])
            min_weights = min(feature["weight"] for feature in reason if feature["weight"])

        for feature in reason:
            id_feature = feature["id"]
            sign = feature["sign"]
            weight = feature["weight"]
            x, y = self.instance_index_to_pixel_position(id_feature - 1, self.image["shape"])
            
            color = (weight / (max_weights - min_weights)) * self.n_colors if with_weights else self.n_colors-1
            if sign:
                self.image_negative[x][y] = color
            else:
                self.image_positive[x][y] = color

        #self.image_negative = numpy.ma.masked_where(self.image_negative < 0.9, self.image_negative)
        #self.image_positive = numpy.ma.masked_where(self.image_positive < 0.9, self.image_positive)
        
        x_1 = pyplot.imshow(PILImage.fromarray(numpy.uint8(self.image_negative)), alpha=0.6, cmap='Reds', vmin=0, vmax=self.n_colors-1, interpolation='None')
        x_2 = pyplot.imshow(PILImage.fromarray(numpy.uint8(self.image_positive)), alpha=0.6, cmap='Blues', vmin=0, vmax=self.n_colors-1, interpolation='None')

        new_image_negative = x_1.make_image(pyplot.gcf().canvas.get_renderer(), unsampled=True)
        new_image_positive = x_2.make_image(pyplot.gcf().canvas.get_renderer(), unsampled=True)
        new_image_negative = PILImage.fromarray(new_image_negative[0])
        new_image_positive = PILImage.fromarray(new_image_positive[0])
            
        fusion = PILImage.blend(new_image_negative, new_image_positive, 0.5)
        if instance is not None:
            array_image = self.instance_to_numpy(instance)
            x_3 = pyplot.imshow(numpy.uint8(array_image), alpha=0.2, cmap='Greys', vmin=0, vmax=255)
            new_image_x_3 = x_3.make_image(pyplot.gcf().canvas.get_renderer(), unsampled=True)
            new_image_x_3 = PILImage.fromarray(new_image_x_3[0])
            fusion = PILImage.blend(fusion, new_image_x_3, 0.4)
        return ImageQt(fusion)
    