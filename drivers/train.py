import onnx
import onnxruntime.training.onnxblock as onnxblock
from onnxruntime.training import artifacts
from onnxruntime.training.api import CheckpointState, Module, Optimizer
from utils import convert_tf_to_onnx
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import evaluate
from onnxruntime_extensions import onnx_op, PyCustomOpDef, get_library_path

@onnx_op(op_type = "QuantDequant",
         inputs =[PyCustomOpDef.dt_float],
         outputs=[PyCustomOpDef.dt_float])
def QuantDequant(data):
    bit_size = 8
    data = np.asarray(data, dtype=np.float32)
    r_min, r_max = np.min(data), np.max(data)

    if r_min == r_max:
        return 1.0, 0
    
    S = (r_max - r_min)/(2**bit_size - 1)
    Z = 0 

    Q = np.clip(np.round(data / S + Z), -2**(bit_size-1), 2**(bit_size-1) - 1)

    return (Q - Z) * S

class MNISTTrainingBlock(onnxblock.TrainingBlock):
    def __init__(self):
        super(MNISTTrainingBlock, self).__init__()
        self.loss = onnxblock.loss.CrossEntropyLoss()

    def build(self, output_name):
        return self.loss(output_name), output_name
    
def create_training_model(untrained_onnx_model):
    # # Create onnx model with loss
    # training_block = MNISTTrainingBlock()
    # for param in untrained_onnx_model.graph.initializer:
    #     print(param.name)
    #     if ("const" in param.name):
    #         training_block.requires_grad(param.name, False)
    #     else:
    #         training_block.requires_grad(param.name, True)

    # # Create training graph and eval graph
    # model_params = None
    # with onnxblock.base(untrained_onnx_model):
    #     _ = training_block(*[output.name for output in untrained_onnx_model.graph.output])
    #     training_model, eval_model = training_block.to_model_proto()
    #     model_params = training_block.parameters()

    # # Create optimizer graph
    # optimizer_block = onnxblock.optim.AdamW()
    # with onnxblock.empty_base() as accessor:
    #     _ = optimizer_block(model_params)
    #     optimizer_model = optimizer_block.to_model_proto()

    # onnxblock.save_checkpoint(training_block.parameters(), "models/train_data/checkpoint")
    # onnx.save(training_model, "models/train_data/training_model.onnx")
    # onnx.save(optimizer_model, "models/train_data/optimizer_model.onnx")
    # onnx.save(eval_model, "models/train_data/eval_model.onnx")

        #     if ("const" in param.name):
    #         training_block.requires_grad(param.name, False)
    #     else:
    #         training_block.requires_grad(param.name, True)

    requires_grad = []
    frozen_params = []
    for initializer in untrained_onnx_model.graph.initializer:
        if "const" in initializer.name and "qdq" in initializer.name:
            frozen_params.append(initializer.name)
        else:
            requires_grad.append(initializer.name)

    artifacts.generate_artifacts(
        onnx_model,
        optimizer=artifacts.OptimType.AdamW,
        loss=artifacts.LossType.CrossEntropyLoss,
        requires_grad=requires_grad,
        frozen_params=frozen_params,
        artifact_directory="models/train_data",
        custom_op_library=get_library_path(),
        additional_output_names=["output"])

    # Create checkpoint state.
    state = CheckpointState.load_checkpoint("models/train_data/checkpoint")

    # Create module.
    training_model = Module("models/train_data/training_model.onnx", state, "models/train_data/eval_model.onnx")

    # Create optimizer.
    optimizer = Optimizer("models/train_data/optimizer_model.onnx", training_model)

    return training_model, optimizer

def load_data(dataset, train_batch_size, test_batch_size):
    def normalize_img(image, label):
        '''Normalizes images: `uint8` -> `float32`.'''
        return tf.cast(image, tf.float32) / 255., label
    
    (ds_train, ds_test), ds_info = tfds.load(
        dataset,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(train_batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(test_batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test

# Perform one training iteration
def train(model, ds_train, epoch):
    model.train()
    losses = []
    for data, label in ds_train:
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2])).astype(np.float32)
        label = np.array(label).astype(np.int64)
        forward_inputs = [data, label]
        
        train_loss, _ = model(*forward_inputs)
        optimizer.step()
        model.lazy_reset_grad()
        losses.append(train_loss)

    print(f'Epoch: {epoch+1},Train Loss: {sum(losses)/len(losses):.4f}')

# Evaluate the model
def test(model, ds_test, epoch):
    model.eval()
    losses = []
    metric = evaluate.load('accuracy')

    for data, label in ds_test:
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2])).astype(np.float32)
        label = np.array(label).astype(np.int64)
        forward_inputs = [data, label]

        test_loss, logits = model(*forward_inputs)
        metric.add_batch(references=label, predictions=np.argmax(logits, axis=1))
        losses.append(test_loss)

    metrics = metric.compute()
    print(f'Epoch: {epoch+1}, Test Loss: {sum(losses)/len(losses):.4f}, Accuracy : {metrics["accuracy"]:.2f}')
    
if __name__ == "__main__":
    # saved_model_path = "models/untrained_model.keras"
    # onnx_model_path = "models/untrained_model.onnx"
    # trained_model_path = "models/trained_model.onnx"

    onnx_model_path = "models/qat_model.onnx"
    trained_model_path = "models/trained_qat_model.onnx"

    # onnx_model = convert_tf_to_onnx(saved_model_path, onnx_model_path)
    onnx_model = onnx.load(onnx_model_path)

    training_model, optimizer = create_training_model(onnx_model)

    dataset = "mnist"
    train_batch_size = 128
    test_batch_size = 128

    ds_train, ds_test = load_data(dataset, train_batch_size, test_batch_size)

    for epoch in range(6):
        train(training_model, ds_train, epoch)
        test(training_model, ds_test, epoch)

    inference_output_names = ["output"]
    training_model.export_model_for_inferencing(trained_model_path, inference_output_names)