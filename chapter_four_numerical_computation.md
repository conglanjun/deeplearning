# 第四章 数值计算
机器学习算法通常需要大量的数值计算。通常是指通过迭代过程更新解的估计值来解决数学问题的算法，而不是通过解析过程推到出公式来提供正确解的方法。常见的操作包括优化和线性方程组的求解。对数字计算机来说实数无法在有限内存下精确表示，因此仅仅是计算涉及实数的函数也是困难的。
## 4.1 上溢和下溢
一种极具毁灭性的舍入是`下溢(underflow)`。当接近零的数被四舍五入为零时发生下溢。
另一个极具破坏的数值错误形式是`上溢(overflow)`。
必须对上溢和下溢数值稳定的一个例子是`softmax函数(softmax function)`。softmax函数经常用于预测与Multinoulli分布相关联的概率，定义为：
$$
softmax(x)_i=\frac{\exp(x_i)}{\sum_{j=1}^n \exp(x_j)}
$$
如果$x_i$等于常数c，所有输出都应该是$\frac{1}{n}$，如果c是很小的负数，$\exp(x_i)$会下溢。意味着softmax分母会变成0，so结果是未定义。当c是非常大的正数时，$\exp(x_i)$的上溢再次使整个表达式未定义。可通过计算softmax(z)解决，其中$z=x-\max x_i$，减去$\max x_i$导致$\exp$的最大值为0，排除上溢的可能性，同时分母有至少有一个是1。排除了下溢导致分母是0的可能性。
```python
import numpy as np
import numpy.linalg as la
x = np.array([1e7, 1e8, 2e5, 2e7])
print(x)
y = np.exp(x) / sum(np.exp(x))
print("上溢：", y)
x = x - np.max(x)
print(x)
y = np.exp(x)/sum(np.exp(x))
print("上溢处理：", y)
------------------
[1.e+07 1.e+08 2.e+05 2.e+07]
上溢： [nan nan nan nan]
[-90000000.         0. -99800000. -80000000.]
上溢处理： [0. 1. 0. 0.]
-------------------
x = np.array([-1e10, -1e9, -2e10, -1e10])
y = np.exp(x)/sum(np.exp(x))
print("下溢：", y)
x = x - np.max(x)
print(x)
y = np.exp(x)/sum(np.exp(x))
print("下溢处理：", y)
print("log softmax(x):", np.log(y))
# 对 log softmax 下溢的处理：
def logsoftmax(x):
    print('x:', x)
    print('sum:', sum(np.exp(x)))
    y = x - np.log(sum(np.exp(x)))
    return y
print("logsoftmax(x):", logsoftmax(x))
-------------------
下溢： [nan nan nan nan]
[-9.0e+09  0.0e+00 -1.9e+10 -9.0e+09]
下溢处理： [0. 1. 0. 0.]
log softmax(x): [-inf   0. -inf -inf]
x: [-9.0e+09  0.0e+00 -1.9e+10 -9.0e+09]
sum: 1.0
logsoftmax(x): [-9.0e+09  0.0e+00 -1.9e+10 -9.0e+09]
```

## 4.2 病态条件

条件数表征函数相对于输入的微小变化而变化的快慢程度。输入被轻微扰动而迅速改变的函数对于科学计算来说可能是有问题的，因为输入中的舍入误差可能导致输出误差的巨大变化。
考虑函数$f(x)=A^{-1}x$当$A\in n\times n$具有特征值分解时，其条件数为：
$$
\max_{i,j} \left| \begin{array}{c} \frac{\lambda_i}{\lambda_j}\end{array} \right|
$$
这是最大和最小特征值的模之比。当该数很大时，矩阵求逆对输入的误差别特敏感。这种敏感性是矩阵本身的固有特性，而不是矩阵求逆期间舍入误差的结果。
## 4.3 基于梯度的优化方法
大多数深度学习的算法都涉及某种形式的优化。优化指的是改变x以最小化或最大化某个函数f(x)的任务。通常以最小化f(x)指代大多数优化问题。
我们把要最小化或最大化的函数称为`目标函数(objective function)`或`准则(criterion)`。当对其进行最小化时称它为`代价函数(cost function)`、`损失函数(loss function)`或`误差函数(error function)`。
通常使用上标$*$表示最小化或最大化函数的x值。如我们记$x^*=arg\min f(x)$。
将x往导数的反方向移动一小步来减小f(x)，这种技术被称为`梯度下降(gradient descent)`。
当$f(x)^{’}=0$，导数无法提供往哪个方向移动的信息。$f(x)^{’}=0$的点称为`临界点(critical point)`或`驻点(stationary point)`。一个`局部极小点(local minimum)`意味着这个点f(x)小于所有临近点。so不能通过移动无穷小的步长来减小f(x)。一个`局部极大点(local maximum)`意味着这个点f(x)大于所有临近点。so不能通过移动无穷小的步长来增大f(x)。有些临界点既不是最小点也不是最大点。这些点被称为`鞍点(saddle point)`。
使f(x)取得绝对的最小值的点是`全局最小点(global minimum)`。函数可能只有一个全局最小点或存在多个全局最小点，可能还存在不是全局最优的局部极小点。深度学习背景下，我们要优化的函数可能含有许多不是最优的局部极小点或者还有很多处于非常平坦的鞍点，尤其是输入是多维的时候，所有这些都将使优化变得困难。so我们通常寻找使f非常小的点，但并不一定是最小。
我们经常最小化具有多维输入的函数：$f:\mathbb{R}^n \rightarrow \mathbb{R}$，为了使“最小化”的概念有意义，输出必须是一维(标量)。
针对具有多维输入的函数，我们需要用到`偏导数(partial derivative)`的概念。偏导数$\frac{\partial}{\partial x_i} f(x)$衡量点x处只有$x_i$增加时f(x)如何变化。`梯度(gradient)`是相对一个向量求导的导数:f的导数是包含所有偏导数的向量，记作$\nabla_x f(x)$。梯度的第i个元素是f关于$x_i$的偏导数。在多维情况下，临界点是梯度中所有元素都为0的点。
在$\mu$(单位向量)方向的`方向导数(directinal derivative)`是函数f在$\mu$方向的斜率。方向导数是函数$f(x+\alpha\mu)$关于$\alpha$的导数(在$\alpha=0$时取得)。使用链式法则，可以看到当$\alpha=0$时，$\frac{\partial}{\partial\alpha}f(x+\alpha\mu)=\mu^T\nabla_xf(x)$。
为了最小化f，我们希望找到使f下降得最快的方向(个人理解：方向导数越小，为负，下降才快。so下面要取方向导数的min)。计算的方向导数：
$$
\min_{\mu,\mu^T\mu=1}\mu^T\nabla_xf(x)\\
=\min_{\mu,\mu^T\mu=1}\parallel\mu\parallel_2\parallel\nabla_xf(x) \parallel_2 \cos\theta
$$
其中$\theta$是$\mu$与梯度的夹角。将$\parallel\mu\parallel_2=1$，代入并忽略与$\mu$无关的项，就能简化得到
$$
\min_{\mu}\cos\theta
$$
在$\mu$与梯度方向相反时取得最小。也就是说，梯度向量指向上坡，负梯度向量指向下坡。在负梯度方向移动可以减小f。这被称为`最速下降法(method of steepest descent)`或`梯度下降(gradient descent)`。最速下降建议新的点为
$$
x^{’}=x-\epsilon\nabla_xf(x)
$$
其中$\epsilon$为`学习率(learning rate)`，是一个确定大小的正标量。可以通过几种不同方式选择$\epsilon$。普遍选择一个小常数。有时我们通过计算，选择使方向导数消失的步长。还有种方法是根据几个$\epsilon$计算$f(x-\epsilon\nabla_xf(x))$，选择其中能产生最小目标值的$\epsilon$。这种策略被称为线搜索。最速下降在梯度的每一个元素为零时收敛(或在实践中很接近零)。在某些情况下，我们也许能够避免运行该迭代算法，通过解方程$\nabla_x f(x)=0$直接跳到临界点。
虽然梯度下降被限制在连续空间中的优化问题，但不断向更好的情况移动一小步的概念可以推广到离散空间。递增带有离散参数的目标函数被称为`爬山(hill climbing)`算法。

### 4.3.1 梯度之上：Jacobian和Hession矩阵
有时需要计算输入和输出都为向量的函数的所有偏导数。包含所有这样偏导数的矩阵称为Jacobian矩阵。具体来说，如果我们有一个函数$f:\mathbb{R}^m-\mathbb{R}^n$，f的Jacobin矩阵$J\in \mathbb{R}^{n\times m}$定义为$J_{i,j}=\frac{\partial}{\partial x_j}f(x)_i$。

补充知识：
Jacobian相对于通用型函数的一阶导数，Hession矩阵是一个$\mathbb{R}^n\rightarrow \mathbb{R}$的函数的二阶导数。本质上说，一个函数对(行)向量求导，本质上还是为向量每个元素进行求导。比如$\mathbb{R}^n\rightarrow \mathbb{R}$的函数$f(\vec{a})$，则其导数(梯度)为$\nabla f=[\frac{\partial f}{\partial a_1}, \frac{\partial f}{\partial a_2}, ... , \frac{\partial f}{\partial a_n}]$，此时一阶导数就变成一个$\mathbb{R}^n\rightarrow \mathbb{R}^n$的函数，对于$\mathbb{R}^m\rightarrow \mathbb{R}^n$的函数，可以看成一个长度为n列向量，对一个长度为m的行向量求偏导。
$$
\frac{\partial \vec{y}}{\partial \vec{a}}=\left[ \begin{array}{c} 
\frac{\partial y_1}{\partial a_1}, ... , \frac{\partial y_1}{\partial a_m}\\
..., ..., ...\\
\frac{\partial y_n}{\partial a_1}, ... , \frac{\partial y_n}{\partial a_m}
\end{array}\right]
$$
有时也对导数的导数感兴趣，即`二阶导数(second derivative)`。例如，有一个函数$f:\mathbb{R}^m\rightarrow\mathbb{R}$，f的一阶导数(关于$x_j$)关于$x_i$的导数记为$\frac{\partial^2}{\partial x_i\partial x_j}f$。在一维情况下，可以将$\frac{\partial^2}{\partial x^2}f$为$f^{’’}(x)$。二阶导数告诉我们，一阶导数将如何随着输入变化而改变。它表示只基于梯度信息的梯度下降步骤是否会如我们预期那样大的改善，二阶导数是曲率的衡量。假设有一个二次函数，如果这样的函数具有零二阶导数，那么就没有曲率。也就是一条完全平坦的线，仅用梯度就可以预测他的值。我们使用沿负梯度方向大小为$\epsilon$的下降步，当梯度是1时，代价函数将下降$\epsilon$。如果二阶导数是负，函数曲线向上凸出，因此代价函数将下降比$\epsilon$多。如果二阶导数是正，函数曲线向下凹，因此代价函数将下降比$\epsilon$少。
当函数有多维输入时，二阶导数也有很多。我们可以将这些导数合并成一个矩阵，称为`Hession矩阵`，Hession矩阵$H(f)(x)$定义为
$$
H(f)(x)_{i, j}=\frac{\partial^2}{\partial x_i\partial x_j}f(x)
$$
Hessian等价于梯度的Jacobian矩阵。
微分算子在任何二阶偏导连续连续的点处可交换，也就是顺序可以互换：
$$
\frac{\partial^2}{\partial x_i \partial x_j}f(x)= \frac{\partial^2}{\partial x_j \partial x_i}f(x)
$$
这意味着$H_i,j=H_j,i$，因此Hession在这些点上是对称的。因为Hession是是对称矩阵，我们可以将其分解成一组实特征值和一组特征向量的正交基，在特定方向$d$上的二阶导数可写成$d^THd$。当$d$是$H$的一个特征向量时，这个方向的二阶导数就是对应的特征值。对于其他的方向$d$，方向二阶导数是所有特征值的加权平均，权重在0和1之间，且与$d$夹角越小的特征向量权重越大。最大特征值确定最大二阶导数，最小特征值确定最小二阶导数。
我们可以通过(方向)二阶导数预期一个梯度下降步骤能表现有多好。我们在当前点$x^{(0)}$处作函数f(x)近似二阶泰勒级数：
$$
f(x)\approx f(x^{(0)})+(x-x^{(0)})^Tg+\frac{1}{2}(x-x^{(0)})^TH(x-x^{(0)})
$$
其中$g$是梯度，$H$是$x^{(0)}$点的Hession。如果使用学习率$\epsilon$，那么新的点x将会是$x^{(0)}-\epsilon g$代入上述的近似，可得：
$$
f(x^{(0)}-\epsilon g)\approx f(x^{(0)})-\epsilon g^Tg+\frac{1}{2}\epsilon^2g^THg
$$
其中有3项：函数原始值、函数斜率导致的预期改善、函数曲率导致的校正。当最后一项太大时，梯度下降实际上是可能向上移动的。当$g^THg$为零或负时，近似的泰勒级数表明增加$\epsilon$将永远使f下降。在实践中，泰勒级数不会在$\epsilon$大的时候也保持准确，so在这种情况下必须采取更启发式的选择。当$g^THg$为正时，通过计算可得，使近似泰勒级数下降最多的最优步长为：
$$
\epsilon^*=\frac{g^Tg}{g^THg}
$$
知识补充：
写一下推到过程，泰勒级数展开式对$\epsilon$求导，令导数为0。
$$
\frac{df(x^{(0)}-\epsilon g)}{d\epsilon}\approx -g^Tg+\epsilon g^THg=0
$$
可以得到上面的$\epsilon^*$。表明最速下降法最优步长，不仅与梯度有关，而且与Hession矩阵有关。
最坏情况下，$g$与$H$最大特征值$\lambda_{max}$对应的特征向量对齐(此处翻译太糟糕，英文版是align with表示对齐，成一条直线的意思。此处最好理解的应该翻译成两个特征向量成一条直线)，最优步长是$\frac{1}{\lambda_{max}}$。我们要最小化的函数能够用二次函数很好的近似的情况下，Hession的特征值决定了学习率的量级。
二阶导数还可以被用来确定一个临界点是否是局部极大点、局部极小点或鞍点。当$f’(x)=0$。而$f’’(x)>0$时，x是一个局部极小点。同样，当$f’(x)=0$。而$f’’(x)<0$时，x是一个局部极大点。这就是`二阶导数测试(second derivative test)`。不幸的是，当$f’’(x)=0$时测试是不确定的。这种情况下，x可以说鞍点或平坦区域的一部分。
在多维情况下，需要检测函数的所有二阶导数。