import dcgan, lsgan, wgan, wgan_gp, ebgan

'''
DCGAN, LSGAN, WGAN, WGAN-GP, EBGAN, BEGAN

Optional:
DRAGAN, CramerGAN

More:
BGAN, MDGAN?
'''

model_zoo = ['DCGAN', 'LSGAN', 'WGAN', 'WGAN-GP', 'EBGAN', 'BEGAN']

def get_model(mtype, name, training):
    model = None
    if mtype == 'DCGAN':
        model = dcgan.DCGAN
    elif mtype == 'LSGAN':
        model = lsgan.LSGAN
    elif mtype == 'WGAN':
        model = wgan.WGAN
    elif mtype == 'WGAN-GP':
        model = wgan_gp.WGAN_GP
    elif mtype == 'EBGAN':
        model = ebgan.EBGAN
    elif mtype == 'BEGAN':
        pass
    else:
        assert False, mtype + ' is not in the model zoo'

    assert model, mtype + ' is work in progress'

    return model(name=name, training=training)


def pprint_args(FLAGS):
    print("\nParameters:")
    for attr, value in sorted(vars(FLAGS).items()):
        print("{}={}".format(attr.upper(), value))
    print("")

