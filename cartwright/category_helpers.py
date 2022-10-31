from cartwright.categories import geos, dates,partial_dates, misc, timespans
import importlib
from inspect import isclass



# TODO: this should be dynamically generated based on e.g. files in the categories directory
def return_all_category_classes_and_labels():
    all_classes={}
    dirs={"geos":geos,"dates":dates,"timespans":timespans,"misc":misc, "partial_dates":partial_dates}
    classes_all={}
    for dir_ in dirs:
        classes_all[dir_]=[x for x in dir(dirs[dir_]) if isclass(getattr(dirs[dir_], x))]

    for dir_ in dirs:

        module = importlib.import_module(f"cartwright.categories.{dir_}")
        for class_name in classes_all.get(dir_):
            if class_name in ["Fake", "Faker", "DateBase","GeoBase", "CategoryBase", "defaultdict","MiscBase", "TimespanBase"]:
                pass
            else:
                try:
                    class_ = getattr(module, class_name)
                    instance = class_()
                    all_classes[instance.return_label()]=instance
                except:
                    print(class_name)

    return all_classes


def generate_label_id(all_labels):

    label2id = {}
    count = 0
    for label in all_labels:
        label2id[label] = count
        count += 1

    return label2id

