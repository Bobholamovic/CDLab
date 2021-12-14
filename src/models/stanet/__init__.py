def create_model(*args, **kwargs):
    from .CDFA_model import CDFAModel
    return CDFAModel(*args, **kwargs)