path = dict(
        server = '/strNet',
        store_loc = 'bulk_dev',
        client = 'MA_Burlington',
        vehicle = '011',
        survey = '20190418_MA_Burlington_Undef_1',
        session = 0,
        environment = 'Training_Data',
)

dir_params = dict(
        training = 'training',
        test = 'test',
        project = '20190701_split_and_cropped_512x512',
        img = 'images',
        mask = 'masks',
        training_size = 0.7,
)

image_params = dict(
        width = 512,
        height = 512,
        channel = 3,
)

u_net_params = dict(
        padding = 'same',
        activation1 = 'relu',
        activation2 = 'sigmoid',
        kernel_down_size = 3,
        kernel_up_size = 2,
        stride = 2,
        pooling = 2,
)

model_params = dict(
        seed_val = 4,
        framework ='keras',
        architecture = 'u-net',
        model_name = 'model_test_2'
)

training_params = dict(
        preproc_id = 1,
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = 'mean_iou',
        model_name = 'model-streetscan-signals-512x512-6.h5',
        patience = 8,
        validation_split = 0.05,
        batch_size = 8,
        epochs = 2,
)

testing_params = dict(
        model_name = 'model-streetscan-signals-512x512-2.h5',
        metrics = 'mean_iou',
)

system_params = dict(
        computer_name = 'test',
        computer_id = 0,
        processor = 'test',
        GPU = 'true',
        memory = 0.0,
        location = 'test',
        activation_date = 'test',
)
