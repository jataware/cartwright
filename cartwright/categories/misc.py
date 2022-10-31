from cartwright.CategoryBases import CategoryBase, MiscBase
import numpy as np


class first_name(MiscBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())


class percent(MiscBase):
    def __init__(self):
        super().__init__()
        self.percent=[
            100,
            1000,
            10000
        ]

    def generate_training_data(self):
        scale = lambda x: x / np.random.choice(self.percent)
        val = str(np.random.choice(scale(np.array(list(range(0, 100))))))
        return self.class_name(), str(val)


class ssn(MiscBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())


class language_name(MiscBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())


class country_name(MiscBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, "country")())


class phone_number(MiscBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())



class zipcode(MiscBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())



class paragraph(MiscBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())


class pyfloat(MiscBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())


class email(MiscBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())



class prefix(MiscBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())


class pystr(MiscBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())



class boolean(MiscBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    # def validate(self, value):
    #     return value.lower() in ('true', 'false', 't', 'f')




class boolean_letter(MiscBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), np.random.choice(['t', 'f','T', 'F'])

    # def validate(self, value):
    #     return value.lower() in ('true', 'false', 't', 'f')


