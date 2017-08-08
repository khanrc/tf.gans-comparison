import dcgan, lsgan


model_zoo = ['DCGAN', 'LSGAN', 'WGAN', 'WGAN-GP', 'BEGAN']

def get_model(mtype, name, input_pipe):
    model = None
    if mtype == 'DCGAN':
        model = dcgan.DCGAN
    elif mtype == 'LSGAN':
        model = lsgan.LSGAN
    elif mtype == 'WGAN':
        pass
    elif mtype == 'WGAN-GP':
        pass
    elif mtype == 'BEGAN':
        pass
    else:
        assert False, mtype + ' is not in the model zoo'

    assert model, mtype + ' is work in progress'

    return model(input_pipe=input_pipe, name=name)


def pprint_args(FLAGS):
    print("\nParameters:")
    for attr, value in sorted(vars(FLAGS).items()):
        print("{}={}".format(attr.upper(), value))
    print("")