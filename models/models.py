import copy


models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model


def make_pretrained(model_spec, args=None, fixed_part=None):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model_pretrained = models[model_spec['name']](**model_args)
    model_pretrained.load_state_dict(model_spec['sd'])
    model = models[model_spec['name']](**model_args)
    if fixed_part is not None:
        if fixed_part == "decoder":
            model.imnet = model_pretrained.imnet
            model.imnet = model.imnet.eval()
            for param in model.imnet.parameters():
                param.requires_grad = False
        elif fixed_part == "encoder":
            model.encoder = model_pretrained.encoder
            model.encoder = model.encoder.eval()
            for param in model.encoder.parameters():
                param.requires_grad = False        
    return model

def make_finetune(model_spec, args=None, fixed_part=None):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model_pretrained = models[model_spec['name']](**model_args)
    model_pretrained.load_state_dict(model_spec['sd'])
    model = models[model_spec['name']](**model_args)
    if fixed_part is not None:
        if fixed_part == "decoder":
            model.encoder = model_pretrained.encoder
            model.imnet = model_pretrained.imnet
            model.imnet = model.imnet.eval()
            for param in model.imnet.parameters():
                param.requires_grad = False
        elif fixed_part == "encoder":
            model.imnet = model_pretrained.imnet
            model.encoder = model_pretrained.encoder
            model.encoder = model.encoder.eval()
            for param in model.encoder.parameters():
                param.requires_grad = False        
    return model
