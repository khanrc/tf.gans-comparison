import dcgan, lsgan, wgan, wgan_gp, ebgan, began, dragan

'''
DCGAN, LSGAN, WGAN, WGAN-GP, EBGAN, BEGAN

Optional:
DRAGAN, CramerGAN

More:
BGAN, MDGAN?

+ SGAN (StackGAN)
'''

model_zoo = ['DCGAN', 'LSGAN', 'WGAN', 'WGAN-GP', 'EBGAN', 'BEGAN', 'DRAGAN']

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
        model = began.BEGAN
    elif mtype == 'DRAGAN':
        model = dragan.DRAGAN
    else:
        assert False, mtype + ' is not in the model zoo'

    assert model, mtype + ' is work in progress'

    return model(name=name, training=training)


def pprint_args(FLAGS):
    print("\nParameters:")
    for attr, value in sorted(vars(FLAGS).items()):
        print("{}={}".format(attr.upper(), value))
    print("")

