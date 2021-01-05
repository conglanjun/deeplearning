## 第三章 概率与信息论
概率论是用来表示不确定性的。直接与频率相关的叫`频率派概率（frequentist probability）`，涉及到确定性水平叫`贝叶斯概率（Bayesian probability）`。   

- **频率学派概率（Frequentist Probability）**：认为概率和事件发生的频率相关。频率学派 - Frequentist - Maximum Likelihood Estimation (MLE，最大似然估计)

- **贝叶斯派概率（Bayesian Probability）**：认为概率是对某件事发生的确定程度，可以理解成是确信的程度。贝叶斯学派 - Bayesian - Maximum A Posteriori (MAP，最大后验估计)

- **随机变量 (Random Variable)**：⼀个可能随机取不同值的变量。例如：抛掷⼀枚硬币，出现正⾯或者反⾯的结果。

扩展资料：

两大学派的争论

抽象一点来讲，频率学派和贝叶斯学派对世界的认知有本质不同：频率学派认为世界是确定的，有一个本体，这个本体的真值是不变的，我们的目标就是要找到这个真值或真值所在的范围；而贝叶斯学派认为世界是不确定的，人们对世界先有一个预判，而后通过观测数据对这个预判做调整，我们的目标是要找到最优的描述这个世界的概率分布。

### 3.2 随机变量
`随机变量（Random variable）`是可以取不同值的变量。
### 3.3 概率分布
`概率分布（probability distrubution）`用来描述变量或一簇随机变量在每一个可能取到的状态的可能性大小。我们描述概率分布的方式取决于取决于随机变量是离散的还是连续的。
#### 3.3.1 离散变量和概率质量函数
离散变量的概率分布可以用`概率质量函数（probability mass function，PMF）`来描述。概率质量函数将将随机变量取得的每个状态映射到取得该变量的概率。明确写出随机变量名$P(x=x)$
概率质量函数可以同时作用于多个随机变量，多个变量的概率分布被称为`联合概率分布（joint probability distribution）`。$P(x=x,y=y)$表示x=x，y=y同时发生的概率简写$P(x,y)$。
PMF必须满足：
- P的定义域是x所有可能状态集合
- $\forall x \in x, 0 \le P(x) \le 1.$不可能发生的事件概率为0。
- $\sum_{x\in x} P(x) =1.$这条性质称为`归一化（normalized）`。例如，离散型随机变量x有k个不同的状态。我们可以假设x是`均匀分布（uniform distribution）`，PMF为：
$$
P(x=x_i)=\frac{1}{k}
$$
#### 3.3.2 连续变量和概率密度函数
研究连续型随机变量用`概率密度函数（probability density function，PDF）`描述概率分布，不用概率质量函数。概率密度函数p满足：
- p的定义域是x所有可能状态集合
- $\forall x \in x, p(x) \ge 0.$，我们不要求$p(x) \le 1$。
- $\int p(x)dx=1.$  

概率密度p(x)没有对特定状态给出概率，它给出了落在面积为$\delta x$的无限小的区域内的概率为$p(x)\delta x$。可以对概率密度函数求积分来获取点集的真实概率质量。单变量例子中，x落在区间$[a, b]$的概率是$\int_{[a, b]} p(x)dx$。
### 3.4 边缘概率
当我们知道了一组变量的联合概率分布，要了解其中一个子集的概率分布。这种定义在子集上的概率分布称为`边缘概率分布（marginal probability distribution）`。  

例如有随机变量x和y，并且知道$P(x,y)$。根据一下`求和法则（sum rule）`计算P(x)：
$$
\forall x \in x,P(x)=\sum_y P(x=x,y=y)
$$
边缘概率来源于手算边缘概率的计算过程。$P(x,y)$每行表示不同的x值，每列表示不同的y值形成的网格，行求和结果P(x)写在每行右边纸边缘处。  

对连续变量要用积分替代求和：
$$
p(x)=\int p(x,y)dy
$$

### 3.5 条件概率
给定其他事件发生时，感兴趣某个事件出现的概率，叫条件概率。给定x=x，y=y发生的条件概率记为$P(y=y|x=x)$。条件概率可以通过下面公式计算：
$$
P(y=y|x=x)=\frac{P(y=y,x=x)}{P(x=x)}
$$
### 3.6 条件概率链式法则
任何多维随机变量的联合概率分布，都可以分解成只有一个变量条件概率相乘形式。叫`链式法则(chain rule)`或`乘法法则(product rule)`。
### 3.7 独立性和条件独立
两个随机变量x和y，如果概率分布可以表示成两个因子乘积形式，并且一个因子只包含x另一个只包含y，则两个随机变量`相互独立(independent)`：
$$
\forall x \in x,y \in y, p(x=x, y=y) = p(x=x)p(y=y)
$$
如果关于x和y的条件概率分布对于z每个值都可以写成乘积形式，两个随机变量x和y在给定随机变量z时是`条件独立的(conditionally independent)：
$$
\forall x \in x,y \in y,z \in z,p(x=x,y=y|z=z)=p(x=x|z=z)p(y=y|z=z)
$$
$x \perp y$表示x和y相互独立，$x \perp y | z$，x和y在给定z时条件独立。
### 3.8 期望、方差和协方差
函数$f (x)$关于某分布$P(x)$的`期望(expectation)`或者`期望值(expected value)`是指，当x由P产生，f作用于x时，$f(x)$的平均值。对于离散型随机变量，可通过求和得到：
$$
E_{x\sim P}[f(x)]=\sum_x P(x)f(x)
$$
对于连续型随机变量可以通过求积分得到：
$$
E_{x\sim P}[f(x)]=\int_x P(x)f(x)
$$
可简写成$E_x[f(x)]$。进一步简写$E[f(x)]$。期望是线性的。  

`方差(variance)`衡量的是对x依据它的概率分布进行采样，随机变量x函数值会呈现多大差异：
$$
Var(f(x))=E[(f(x)-E(f(x))^2]
$$
当方差很小时f(x)值形成的簇比较接近期望值。方差的平方根称为`标准差(standard variance)`。  

`协方差(covariance)`两个变量线性相关的强度以及这些变量的尺度：
$$
Cov(f(x),g(y))=E[(f(x)-E[f(x)])(g(y)-E[g(y))]
$$
协方差用于衡量两个变量的总体误差，方差是协方差一种特殊情况。公式上看协方差表示的是两个变量总体误差的期望。变化趋势一致则协方差是正。`相关系数(correlation)`将每个变量的值归一化，为了只衡量变量的相关性而不受变量尺度大小影响。  

独立性比零协方差的要求更强，独立性排除了非线性关系。  

随机向量$x\in \mathbb {R}^n$`协方差矩阵(covariance matrix)`是$n\times n$的矩阵。满足：
$$
Cov(x)_{i,j}=Cov(x_i,x_j)
$$
协方差矩阵对角元素是方差：
$$
Cov(x_i,x_i)=Var(x_i)
$$
### 3.9 常用概率分布
#### 3.9.1 Bernouli分布
`Bernoli分布(Bernoli distribution)`是单个二值随机变量分布。它由单个参数$\phi \in [0, 1]$控制，$\phi$给出了随机变量等于1的概率。具有如下性质：
$$
P(\mathrm {x}=1)=\phi \\
P(\mathrm {x}=0)=1-\phi \\
P(\mathrm {x}=x)=\phi ^x (1-\phi)^{1-x} \\
\mathbb {E}_{\mathrm {x}} [\mathrm {x}]=\phi \\
Var_{\mathrm {x}} (\mathrm {x})=\phi (1-\phi)
$$
#### 3.9.2 Multinoulli分布
`Multinoulli分布(Multinoulli distribution)`或者`范畴分布(categorical distrubution)`是指在具有$k$个不同状态的单个离散型随机变量上的分布，其中$k$是一个有限值。Multinoulli分布由向量$p \in [0, 1]^{k-1}$参数化，其中每一个分量$p_i$表示第$i$个状态的概率。最后第$k$个状态概率可以通过$1-1^Tp$给出。(个人理解：这里乘1矩阵是为了让p中所有概率值相加)必须限制$1^Tp < 1$。Multinoulli分布经常用来表示对象分类的分布。Multinoulli分布是`多项式分布(multinomial distribution)`的一个特例。  

`二项分布(n重Bernoulli分布)(binomial distribution)`描述N次独立Bernoulli实验中有m次成功(即x=1)的概率，其中每次Bernoulli实验成功的