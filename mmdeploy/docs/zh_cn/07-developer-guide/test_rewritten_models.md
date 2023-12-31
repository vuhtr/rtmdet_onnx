# 测试模型重写

模型 [rewriter](support_new_model.md) 完成后，还需完成对应测试用例，以验证重写是否生效。通常我们需要对比原始模型和重写后的输出。原始模型输出可以调用模型的 forward 函数直接获取，而生成重写模型输出的方法取决于重写的复杂性。

## 测试简单的重写

如果对模型的更改很小（例如，仅更改一个或两个变量且无副作用），则可为重写函数/模块构造输入，在`RewriteContext`中运行推理并检查结果。

```python
# mmpretrain.models.classfiers.base.py
class BaseClassifier(BaseModule, metaclass=ABCMeta):
    def forward(self, img, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)

# Custom rewritten function
@FUNCTION_REWRITER.register_rewriter(
    'mmpretrain.models.classifiers.BaseClassifier.forward', backend='default')
def forward_of_base_classifier(self, img, *args, **kwargs):
    """Rewrite `forward` for default backend."""
    return self.simple_test(img, {})
```

在示例中，我们仅更改 forward 函数。我们可以通过编写以下函数来测试这个重写：

```python
def test_baseclassfier_forward():
    input = torch.rand(1)
    from mmpretrain.models.classifiers import BaseClassifier
    class DummyClassifier(BaseClassifier):

        def __init__(self, init_cfg=None):
            super().__init__(init_cfg=init_cfg)

        def extract_feat(self, imgs):
            pass

        def forward_train(self, imgs):
            return 'train'

        def simple_test(self, img, tmp, **kwargs):
            return 'simple_test'

    model = DummyClassifier().eval()

    model_output = model(input)
    with RewriterContext(cfg=dict()), torch.no_grad():
        backend_output = model(input)

    assert model_output == 'train'
    assert backend_output == 'simple_test'
```

在这个测试函数中，我们构造派生类 `BaseClassifier` 来测试重写能否工作。通过直接调用`model(input)`来获得原始输出，并通过在`RewriteContext`中调用`model(input)`来获取重写的输出。最后断检查输出。

## 测试复杂重写

有时我们可能会对原始模型函数进行重大更改（例如，消除分支语句以生成正确的计算图）。即使运行在Python中的重写模型的输出是正确的，我们也不能保证重写的模型可以在后端按预期工作。因此，我们需要在后端测试重写的模型。

```python
# Custom rewritten function
@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.segmentors.BaseSegmentor.forward')
def base_segmentor__forward(self, img, img_metas=None, **kwargs):
    ctx = FUNCTION_REWRITER.get_context()
    if img_metas is None:
        img_metas = {}
    assert isinstance(img_metas, dict)
    assert isinstance(img, torch.Tensor)

    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    img_shape = img.shape[2:]
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]
    img_metas['img_shape'] = img_shape
    return self.simple_test(img, img_metas, **kwargs)

```

此重写函数的行为很复杂，我们应该按如下方式测试它：

```python
def test_basesegmentor_forward():
    from mmdeploy.utils.test import (WrapModel, get_model_outputs,
                                    get_rewrite_outputs)

    segmentor = get_model()
    segmentor.cpu().eval()

    # Prepare data
    # ...

    # Get the outputs of original model
    model_inputs = {
        'img': [imgs],
        'img_metas': [img_metas],
        'return_loss': False
    }
    model_outputs = get_model_outputs(segmentor, 'forward', model_inputs)

    # Get the outputs of rewritten model
    wrapped_model = WrapModel(segmentor, 'forward', img_metas = None, return_loss = False)
    rewrite_inputs = {'img': imgs}
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    if is_backend_output:
        # If the backend plugins have been installed, the rewrite outputs are
        # generated by backend.
        rewrite_outputs = torch.tensor(rewrite_outputs)
        model_outputs = torch.tensor(model_outputs)
        model_outputs = model_outputs.unsqueeze(0).unsqueeze(0)
        assert torch.allclose(rewrite_outputs, model_outputs)
    else:
        # Otherwise, the outputs are generated by python.
        assert rewrite_outputs is not None
```

我们已经提供了一些使用函数做测试，例如可以先 build 模型，用  `get_model_outputs` 获取原始输出；然后用`WrapModel` 包装重写函数，使用`get_rewrite_outputs` 获取结果。这个例子里会返回输出内容和是否来自后端两个结果。

因为我们也不确定用户是否正确安装后端，所以得检查结果来自 Python 还是真实后端推理结果。单元测试必须涵盖这两种结果，最后用`torch.allclose` 对比两种结果的差异。

API 文档中有测试用例完整用法。
