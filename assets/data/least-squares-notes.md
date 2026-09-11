

> 订正版。原笔记的主线推导正确,本版修正了记号不一致、`A^TA` 结构的解释错误,
> 并补上了两个缺失的前提(为什么是最小值、什么时候有唯一解)。

---

## 0.  记号约定

全文统一使用**降幂**排列,与 `numpy.polyfit` 的返回顺序一致。

二次拟合的多项式写作

$$p(x)=c_0x^2+c_1x+c_2$$

系数向量

$$c=\begin{bmatrix}c_0\\c_1\\c_2\end{bmatrix}$$

即 $c_0$ 配 $x^2$,$c_2$ 是常数项。

> ⭕️ **原笔记的坑**:开头定义 $p(x)=c_0+c_1x+c_2x^2$(升幂),但矩阵 $A$ 的列是
> $[x^2,\ x,\ 1]$(降幂),两者放在一起 $c_0$ 就错位了。也不再用 $a,b,d$ ——
> 用 $d$ 表示常数项容易和别的东西混,统一成带下标的 $c_i$ 更省心。

一般地,$m$ 次拟合:

$$p(x)=c_0x^m+c_1x^{m-1}+\cdots+c_{m-1}x+c_m$$

有 $m+1$ 个待求系数。

---

## 1.  问题:用多项式拟合数据

### 1.1 设计矩阵

给定 $n$ 组实测数据 $(x_i,y_i)$,把每个 $x_i$ 的各次幂排成一行:

$$A=\begin{bmatrix}
x_1^m & x_1^{m-1} & \cdots & x_1 & 1\\
x_2^m & x_2^{m-1} & \cdots & x_2 & 1\\
\vdots & \vdots & & \vdots & \vdots\\
x_n^m & x_n^{m-1} & \cdots & x_n & 1
\end{bmatrix}_{n\times(m+1)}$$

**术语说明**:这个 $n\times(m+1)$ 的高瘦矩阵,严格讲叫**设计矩阵**(design matrix)。
"范德蒙德矩阵"通常特指**方阵**版本($n=m+1$),那个漂亮的行列式公式

$$\det V=\prod_{i<j}(x_j-x_i)$$

只对方阵成立(且对应升幂列序;列序颠倒会差一个符号)。拟合语境下把 $A$ 叫作
"范德蒙德型矩阵"是常见叫法,但要知道两者的区别。

### 1.2 $Ac$ 是什么

$$Ac=\begin{bmatrix}
x_1^2 & x_1 & 1\\
x_2^2 & x_2 & 1\\
x_3^2 & x_3 & 1
\end{bmatrix}
\begin{bmatrix}c_0\\c_1\\c_2\end{bmatrix}
=\begin{bmatrix}
c_0x_1^2+c_1x_1+c_2\\
c_0x_2^2+c_1x_2+c_2\\
c_0x_3^2+c_1x_3+c_2
\end{bmatrix}
=\begin{bmatrix}\hat y_1\\\hat y_2\\\hat y_3\end{bmatrix}$$

🍏 **关键**:$A$ 的**每一行**与 $c$ 做一次点积,得到**一个点的预测值**。

$$[\,x_i^2,\ x_i,\ 1\,]\begin{bmatrix}c_0\\c_1\\c_2\end{bmatrix}=c_0x_i^2+c_1x_i+c_2=\hat y_i$$

所以"矩阵乘法还原方程组"的机制,就是**逐行做点积**。

各部分的含义:

| 符号 | 含义 | 尺寸 |
|---|---|---|
| $A$ | 由所有 $x$ 的各次幂构成的表 | $n\times(m+1)$ |
| $c$ | 待求的多项式系数 | $(m+1)\times 1$ |
| $Ac$ | 按这组系数算出的**全部预测值** | $n\times 1$ |
| $y$ | 真实观测值 | $n\times 1$ |

### 1.3 超定:为什么写 $\approx$ 而不是 $=$

$$A_{n\times(m+1)}\;c_{(m+1)\times1}\;\approx\;y_{n\times1}$$

当 $n>m+1$(点数多于系数)时,这是**超定方程组**(overdetermined system)——
方程比未知数多。

⭕️ **超定方程组通常无解**。61 个点想让一条抛物线全都精确穿过,除非数据本身就
落在一条抛物线上,否则做不到。

于是问题从"求解"变成"**求最接近**"。这就是最小二乘的出发点。

> 🔴 对照:如果 $n=m+1$(点数 = 系数个数)且 $x$ 两两不同,$A$ 是可逆方阵,
> $Ac=y$ **有唯一精确解**,残差恰好为零。那叫**插值**,不叫拟合。
> 本笔记第 9 节的例子正是这种情况,要留意。

---

## 2.  目标函数:最小二乘

> ⭕️ 这一节在原笔记里排在两次推导之后。它其实是**整篇的地基**,应该先讲清楚:
> 先知道"要最小化什么",再看两条路怎么走到同一个终点。

### 2.1 范数

先定义**残差向量**:

$$r=y-Ac=\begin{bmatrix}
y_1-(c_0x_1^2+c_1x_1+c_2)\\
y_2-(c_0x_2^2+c_1x_2+c_2)\\
\vdots\\
y_n-(c_0x_n^2+c_1x_n+c_2)
\end{bmatrix}$$

> 🔴 **符号统一**:全文一律用 $r=y-Ac$。原笔记两处定义相反($y-Ac$ 与 $Ac-y$),
> 平方后结果一样,但同一份笔记里应当统一。

🍏 **小概念:范数(norm)**

- 单竖线 $|x|$ 是绝对值,用于**标量**
- 双竖线 $\|v\|$ 是范数,用于**向量**,表示"长度"

下标区分不同的范数:

| 记号 | 名称 | 对应的拟合方法 |
|---|---|---|
| $\|r\|_2$ | 欧几里得范数(默认) | **最小二乘** |
| $\|r\|_1$ | 各分量绝对值之和 | 最小绝对偏差回归,抗离群值更强 |
| $\|r\|_\infty$ | 最大分量 | minimax(切比雪夫)拟合 |

不写下标时默认是 2-范数。

### 2.2 目标:让残差向量最短

$$\boxed{\min_c\ \|Ac-y\|^2}$$

展开:$r$ 是 $n$ 维向量,它的欧几里得长度是

$$\|r\|=\sqrt{r_1^2+r_2^2+\cdots+r_n^2}$$

平方后

$$\|r\|^2=\sum_{i=1}^n r_i^2$$

⭕️ **这正是误差平方和 $E(c)$。** 所以两种写法完全等价:

$$\min_c\|Ac-y\|^2
\quad\Longleftrightarrow\quad
\min_{c_0,c_1,c_2}\sum_{i=1}^n\bigl[y_i-(c_0x_i^2+c_1x_i+c_2)\bigr]^2$$

前者简洁,也更容易看出几何含义(见第 5 节);后者便于逐项求导(见第 3 节)。

**为什么是平方,不是绝对值?**

1. 平方可导,绝对值在零点不可导,求导法直接失效
2. 平方使目标函数成为二次型,有闭式解
3. 统计上,若噪声服从高斯分布,最小二乘等价于极大似然估计

代价是对离群点敏感——一个偏离很远的点,平方后权重极大。这时才考虑 $\|\cdot\|_1$。

---

## 3.  视角一:微积分(逐分量偏导)

$$E(c_0,c_1,c_2)=\sum_{i=1}^n\bigl[y_i-(c_0x_i^2+c_1x_i+c_2)\bigr]^2$$

`polyfit` 做的事:找 $c_0,c_1,c_2$ 使这个数最小。

**思路**:对三个未知数分别求偏导,并令导数为零。

$$\frac{\partial E}{\partial c_0}=0,\qquad
\frac{\partial E}{\partial c_1}=0,\qquad
\frac{\partial E}{\partial c_2}=0$$

### 3.1 逐个求导

记残差 $r_i=y_i-(c_0x_i^2+c_1x_i+c_2)$,则 $E=\sum_i r_i^2$。用链式法则:

$$\frac{\partial E}{\partial c_0}=\sum_i 2r_i\frac{\partial r_i}{\partial c_0}$$

**对 $c_0$**:$\dfrac{\partial r_i}{\partial c_0}=-x_i^2$,所以

$$\frac{\partial E}{\partial c_0}=-2\sum_i x_i^2\bigl[y_i-(c_0x_i^2+c_1x_i+c_2)\bigr]=0$$

公共因子 $-2$ 可以消掉:

$$✅\quad \sum_i x_i^2\bigl[y_i-(c_0x_i^2+c_1x_i+c_2)\bigr]=0$$

**对 $c_1$**:$\dfrac{\partial r_i}{\partial c_1}=-x_i$,同理

$$✅\quad \sum_i x_i\bigl[y_i-(c_0x_i^2+c_1x_i+c_2)\bigr]=0$$

**对 $c_2$**:$\dfrac{\partial r_i}{\partial c_2}=-1$,同理

$$✅\quad \sum_i \bigl[y_i-(c_0x_i^2+c_1x_i+c_2)\bigr]=0$$

### 3.2 三个条件的含义

⭕️ 这三个方程分别来自对三个参数求偏导。它们说的是:**在最优点,残差向量与
$A$ 的三列($x_i^2$ 列、$x_i$ 列、常数 1 列)都正交**。

这正是 $A^T(y-Ac)=0$ 的分量形式。第 5 节会看到,这就是"正交投影"的代数表达。

### 3.3 整理成矩阵

把含 $c$ 的项留在左边、其余移到右边:

$$\begin{bmatrix}
\sum x_i^4 & \sum x_i^3 & \sum x_i^2\\
\sum x_i^3 & \sum x_i^2 & \sum x_i\\
\sum x_i^2 & \sum x_i & n
\end{bmatrix}
\begin{bmatrix}c_0\\c_1\\c_2\end{bmatrix}
=\begin{bmatrix}
\sum x_i^2y_i\\
\sum x_iy_i\\
\sum y_i
\end{bmatrix}$$

- 这组方程叫**正规方程组**(normal equations)
- 左边的系数方阵就是 $A^{\mathsf T}A$(下一节会证)
- $A^{\mathsf T}A$ 也叫 **Gram 矩阵**(格拉姆矩阵),因为它的元素是 $A$ 各列的两两点积

🟢 到这一步,它就是一个普通的**三元一次方程组**。解出来就是 $c$。

---

## 4.  视角二:线性代数(矩阵求导)

同样的结果,换成矩阵语言重推一遍。

### 4.1 转置规则速查

$$
(P\pm Q)^{\mathsf T}=P^{\mathsf T}\pm Q^{\mathsf T},\qquad
(PQ)^{\mathsf T}=Q^{\mathsf T}P^{\mathsf T},\qquad
(PQR)^{\mathsf T}=R^{\mathsf T}Q^{\mathsf T}P^{\mathsf T},\qquad
(P^{\mathsf T})^{\mathsf T}=P
$$

注意 $(PQ)^{\mathsf T}$ 要**反序**。

### 4.2 向量长度的矩阵写法

对任意列向量 $v$:

$$\|v\|^2=v^{\mathsf T}v$$

含义:向量长度的平方,等于它与自身的点积;写成矩阵乘法就是"转置乘自己"。

⭕️ **顺序不能反**:

- $v^{\mathsf T}v$ 是 $1\times n$ 乘 $n\times 1$ = **标量**(点积)
- $vv^{\mathsf T}$ 是 $n\times 1$ 乘 $1\times n$ = **$n\times n$ 矩阵**(外积),完全不同的东西

### 4.3 展开目标函数

$$E(c)=\|y-Ac\|^2=(y-Ac)^{\mathsf T}(y-Ac)$$

先算转置:

$$(y-Ac)^{\mathsf T}=y^{\mathsf T}-(Ac)^{\mathsf T}=y^{\mathsf T}-c^{\mathsf T}A^{\mathsf T}$$

代回,像普通代数一样展开四项:

$$E(c)=(y^{\mathsf T}-c^{\mathsf T}A^{\mathsf T})(y-Ac)
=y^{\mathsf T}y-y^{\mathsf T}Ac-c^{\mathsf T}A^{\mathsf T}y+c^{\mathsf T}A^{\mathsf T}Ac$$

### 4.4 中间两项其实是同一个数

要证:$y^{\mathsf T}Ac=c^{\mathsf T}A^{\mathsf T}y$。

> 🔴 **原笔记的坑**:这里令 $A=y^{\mathsf T},B=A,C=c$ ——把 $A$ 同时当占位符和
> 真实矩阵用,极易看晕。改用 $P,Q,R$。

令 $P=y^{\mathsf T},\ Q=A,\ R=c$,由三重转置规则:

$$(y^{\mathsf T}Ac)^{\mathsf T}=c^{\mathsf T}A^{\mathsf T}(y^{\mathsf T})^{\mathsf T}=c^{\mathsf T}A^{\mathsf T}y$$

另一方面,$y^{\mathsf T}Ac$ 的尺寸是 $(1\times n)(n\times k)(k\times 1)=1\times 1$,
**是一个数**。数的转置等于它自己:

$$(y^{\mathsf T}Ac)^{\mathsf T}=y^{\mathsf T}Ac$$

两句合起来:

$$\boxed{y^{\mathsf T}Ac=c^{\mathsf T}A^{\mathsf T}y}$$

于是中间两项合并:

$$-y^{\mathsf T}Ac-c^{\mathsf T}A^{\mathsf T}y=-2c^{\mathsf T}A^{\mathsf T}y$$

最终

$$\boxed{E(c)=y^{\mathsf T}y-2c^{\mathsf T}A^{\mathsf T}y+c^{\mathsf T}A^{\mathsf T}Ac}$$

这是关于 $c$ 的**二次型**,形式上就是 $\text{常数}-2\times\text{一次项}+\text{二次项}$。

### 4.5 求导

把 $y$ 和 $A$ 看成常量,只有 $c$ 是变量。用到两条矩阵求导规则:

$$\frac{\partial}{\partial c}\bigl(c^{\mathsf T}b\bigr)=b,
\qquad
\frac{\partial}{\partial c}\bigl(c^{\mathsf T}Mc\bigr)=(M+M^{\mathsf T})c
\;\xrightarrow[\ M\ \text{对称}\ ]{}\;2Mc$$

> ⭕️ **第二条依赖 $M$ 对称**。这里 $M=A^{\mathsf T}A$,而
> $(A^{\mathsf T}A)^{\mathsf T}=A^{\mathsf T}(A^{\mathsf T})^{\mathsf T}=A^{\mathsf T}A$,
> 确实对称,规则可用。这个对称性后面第 8 节还要用到。

逐项:

- $\dfrac{\partial}{\partial c}(y^{\mathsf T}y)=0$(不含 $c$)
- $\dfrac{\partial}{\partial c}(-2c^{\mathsf T}A^{\mathsf T}y)=-2A^{\mathsf T}y$
- $\dfrac{\partial}{\partial c}(c^{\mathsf T}A^{\mathsf T}Ac)=2A^{\mathsf T}Ac$

合起来:

$$\frac{\partial E}{\partial c}=-2A^{\mathsf T}y+2A^{\mathsf T}Ac$$

令其为零,两边除以 2,移项:

$$\boxed{A^{\mathsf T}Ac=A^{\mathsf T}y}$$

### 4.6 验证:$A^{\mathsf T}A$ 确实是第 3 节那个矩阵

$$A^{\mathsf T}=\begin{bmatrix}
x_1^2 & x_2^2 & \cdots & x_n^2\\
x_1 & x_2 & \cdots & x_n\\
1 & 1 & \cdots & 1
\end{bmatrix}_{3\times n}$$

$$A^{\mathsf T}A
=\underbrace{\begin{bmatrix}
x_1^2 & \cdots & x_n^2\\
x_1 & \cdots & x_n\\
1 & \cdots & 1
\end{bmatrix}}_{3\times n}
\cdot
\underbrace{\begin{bmatrix}
x_1^2 & x_1 & 1\\
\vdots & \vdots & \vdots\\
x_n^2 & x_n & 1
\end{bmatrix}}_{n\times 3}
=\underbrace{\begin{bmatrix}
\sum x_i^4 & \sum x_i^3 & \sum x_i^2\\
\sum x_i^3 & \sum x_i^2 & \sum x_i\\
\sum x_i^2 & \sum x_i & n
\end{bmatrix}}_{3\times 3}$$

$$A^{\mathsf T}y
=\begin{bmatrix}
x_1^2 & \cdots & x_n^2\\
x_1 & \cdots & x_n\\
1 & \cdots & 1
\end{bmatrix}
\begin{bmatrix}y_1\\\vdots\\y_n\end{bmatrix}
=\begin{bmatrix}
\sum x_i^2y_i\\
\sum x_iy_i\\
\sum y_i
\end{bmatrix}$$

🟢 与第 3 节微积分路线得到的完全一致。**两条路,同一个终点。**

---

## 5.  视角三:几何(正交投影)

这是最直观的一个。

$A$ 的 $m+1$ 个列向量,在 $n$ 维空间里张成一个 $(m+1)$ 维**子空间**(列空间
$\mathrm{Col}(A)$)。例如 61 个点做二次拟合,就是 61 维空间里的一个 3 维子空间。

- $Ac$ 是这三列的线性组合,**只能落在这个子空间里**
- $y$ 一般**在子空间外面**

问"哪个 $Ac$ 离 $y$ 最近",答案是:$y$ 在该子空间上的**正交投影**。

正交意味着残差 $r=y-Ac$ 垂直于子空间,也就是垂直于 $A$ 的每一列:

$$A^{\mathsf T}(y-Ac)=0$$

展开即

$$A^{\mathsf T}y-A^{\mathsf T}Ac=0
\quad\Longrightarrow\quad
\boxed{A^{\mathsf T}Ac=A^{\mathsf T}y}$$

⭕️ 三个视角在此汇合:

| 视角 | 说法 |
|---|---|
| 微积分 | 误差平方和的偏导为零 |
| 线性代数 | 二次型的梯度为零 |
| 几何 | 残差与列空间正交 |

三句话说的是同一件事。

---

## 6.  补上两个前提

> ⭕️ 这一节是原笔记**缺失的部分**。求导令零只是必要条件,还差两块。

### 6.1 为什么驻点一定是最小值

令导数为零只保证是**驻点**,原则上可能是极大值或鞍点。需要看二阶信息。

$E(c)$ 的 Hessian 矩阵是

$$H=\frac{\partial^2E}{\partial c\,\partial c^{\mathsf T}}=2A^{\mathsf T}A$$

而对任意向量 $v$:

$$v^{\mathsf T}(A^{\mathsf T}A)v=(Av)^{\mathsf T}(Av)=\|Av\|^2\ \ge\ 0$$

所以 $A^{\mathsf T}A$ **半正定**,$E(c)$ 是**凸函数**,驻点即全局最小值。

进一步,当 $A$ 列满秩时,$Av=0\Rightarrow v=0$,故 $\|Av\|^2>0$($v\neq0$),
$A^{\mathsf T}A$ **正定**,$E$ **严格凸**,最小点**唯一**。

✅ 这就是为什么最小二乘"求导令零"可以直接下结论,不用讨论极大极小——
目标函数是凸的,不存在别的驻点。

### 6.2 什么时候有唯一解

$A^{\mathsf T}Ac=A^{\mathsf T}y$ 有唯一解 $\iff$ $A^{\mathsf T}A$ 可逆
$\iff$ **$A$ 列满秩**。

具体到多项式拟合,条件是:

> ⭕️ **互不相同的 $x$ 值至少有 $m+1$ 个。**

反例:你采了 61 个点,但只在 2 个角度上反复测量。数据再多,$A$ 的秩也只有 2,
$A^{\mathsf T}A$ 奇异,二次拟合无唯一解。**点多 ≠ 信息多**。

一个有用的恒等式(方阵情形):

$$\det(A^{\mathsf T}A)=\det(A^{\mathsf T})\det(A)=\bigl(\det A\bigr)^2$$

可以用来交叉验算(第 9 节会用到)。

---

## 7.  $A^{\mathsf T}$ 干了什么

$$\underbrace{\begin{bmatrix}
x_1^2 & x_2^2 & \cdots & x_n^2\\
x_1 & x_2 & \cdots & x_n\\
1 & 1 & \cdots & 1
\end{bmatrix}}_{A^{\mathsf T}\ (3\times n)}
\cdot
\underbrace{\begin{bmatrix}y_1\\y_2\\\vdots\\y_n\end{bmatrix}}_{y\ (n\times1)}
=
\underbrace{\begin{bmatrix}
\sum x_i^2y_i\\
\sum x_iy_i\\
\sum y_i
\end{bmatrix}}_{A^{\mathsf T}y\ (3\times1)}$$

⭕️ **左乘 $A^{\mathsf T}$**,把"$n$ 个方程、3 个未知数、通常无解"的超定方程组,
压缩成了一个 **3×3 的三元一次方程组**,高斯消元十几次运算就解完。

> 🔴 措辞更正:是"**左乘** $A^{\mathsf T}$",不是"$A$ 通过转置"。$A$ 本身没变。

**不管原始数据是 61 个点还是 61 万个点,到这一步都是同一个 3×3 方程组。**

> 注意:$n$ 并没有"消失"——它就坐在 $A^{\mathsf T}A$ 的右下角
> ($\sum_i 1 = n$)。消失的是**矩阵的尺寸对 $n$ 的依赖**。

### 7.1 计算量

| 步骤 | 代价 |
|---|---|
| 构造 $A^{\mathsf T}A$ | 5 个独立求和,每个 $n$ 项 → $O(n)$ |
| 构造 $A^{\mathsf T}y$ | 3 个求和,每个 $n$ 项 → $O(n)$ |
| 解 3×3 方程组 | 常数级,**与 $n$ 无关** |

### 7.2 核心洞察:充分统计量

🔴 **任意多个点的全部信息,被压缩进了 8 个数**(二次拟合):

- 5 个 $\sum x^k$,$k=0,1,2,3,4$(其中 $\sum x^0=n$)
- 3 个 $\sum x^ky$,$k=0,1,2$

**这一步之后,原始数据就可以扔掉了。** 这在统计学里叫**充分统计量**
(sufficient statistics)——它把样本对参数估计的全部信息都装下了。

一般地,$m$ 次拟合需要 $(2m+1)+(m+1)=3m+2$ 个数。

| 次数 $m$ | $A^{\mathsf T}A$ 的求和数 | $A^{\mathsf T}y$ 的求和数 | 合计 |
|---|---|---|---|
| 1(直线) | 3 | 2 | **5** |
| 2(抛物线) | 5 | 3 | **8** |
| 3(三次) | 7 | 4 | **11** |

### 7.3 两个实际推论

**内存**:不用存 61 个点,只维护 8 个累加器。数据一个一个来,边来边加。

**在线更新**:新来一个点,8 个累加器各加一项,重新解一次 3×3 方程组,
就得到更新后的系数。

> 🔴 **术语更正**:这个做法准确地说是「**正规方程的在线累加**」。
> 标准的**递推最小二乘(RLS)** 特指用 Sherman–Morrison 公式**增量更新
> $(A^{\mathsf T}A)^{-1}$**、每步不重新求解的那一套。
> 对 3×3 来说,累加器版本更简单也更稳,推荐用它。

🍏 **ESP32 应用**:完全跑得动。转一圈边转边累加 8 个 `float`(或 `double`,
见下方警告),最后自己解出系数存进 flash,不用连电脑做标定。

> ⚠️ **数值警告**:$\sum x^4$ 增长很快。若 $x$ 取值范围大(比如 0~1000),
> $\sum x^4$ 可能上到 $10^{14}$,单精度 `float` 只有约 7 位有效数字,会溢出精度。
> **对策**:先把 $x$ 平移缩放到 $[-1,1]$ 再拟合,最后把系数换算回去;
> 累加器一律用 `double`。这个技巧叫**中心化/标准化**,`numpy` 内部也在做。

---

## 8.  $A^{\mathsf T}A$ 的结构

### 8.1 三个小概念

以 3×3 为例,元素位置编号:

```
(1,1) (1,2) (1,3)
(2,1) (2,2) (2,3)
(3,1) (3,2) (3,3)
```

- **主对角线**:行号 = 列号,即 (1,1)、(2,2)、(3,3),从左上到右下那条斜线
- **上三角**:主对角线 + **右上方**

```
 ■  ■  ■     ← (1,1) (1,2) (1,3)
 ·  ■  ■     ←       (2,2) (2,3)
 ·  ·  ■     ←             (3,3)
```

  数一数:3 + 2 + 1 = **6 个**

- **下三角**:主对角线 + **左下方**,同样 6 个

> 🔴 **原笔记的错误**:把下三角写成"右下部分的小三角"。下三角是**左下**。
> 而且严格上三角和严格下三角**元素个数相同**(各 3 个),不存在"大/小"之分。

### 8.2 性质一:对称

$$(A^{\mathsf T}A)^{\mathsf T}=A^{\mathsf T}(A^{\mathsf T})^{\mathsf T}=A^{\mathsf T}A$$

所以 $M_{ij}=M_{ji}$,沿对角线对折两半重合。

**更直观的理解**:$(A^{\mathsf T}A)_{ij}$ = ($A^{\mathsf T}$ 的第 $i$ 行)·($A$ 的第 $j$ 列)
= (**$A$ 的第 $i$ 列**)·(**$A$ 的第 $j$ 列**)。列 $i$ 点乘列 $j$ 当然等于列 $j$ 点乘列 $i$。

这也是它叫 **Gram 矩阵**的原因:各列两两点积排成的表。

✌🏻 **免费的检查手段**:算完 $A^{\mathsf T}A$ 先看是否对称。不对称一定算错了。

于是 6 个上三角元素之外,下面 3 个**照抄即可,不用算**。

### 8.3 性质二:Hankel(这才是"5 个数"的真正原因)

> 🔴 **原笔记这里理由错了**,说是"减去已知的 $n$"。实际原因如下。

$A$ 的第 $i$ 列是 $x^{m-i+1}$,所以

$$(A^{\mathsf T}A)_{ij}=\sum_k x_k^{(m-i+1)+(m-j+1)}=\sum_k x_k^{\,2m+2-i-j}$$

⭕️ **指数只依赖 $i+j$**,与 $i,j$ 各自是多少无关。这意味着矩阵沿着
**每条反对角线(从右上到左下)取值完全相同**。这种矩阵叫 **Hankel 矩阵**。

$$A^{\mathsf T}A=\begin{bmatrix}
\sum x^4 & \sum x^3 & \sum x^2\\
\sum x^3 & \sum x^2 & \sum x\\
\sum x^2 & \sum x & n
\end{bmatrix}$$

看反对角线:

- (1,3)、(2,2)、(3,1) —— 三个都是 $\sum x^2$ ← **注意 (2,2) 也在里面**
- (1,2)、(2,1) —— 都是 $\sum x^3$
- (2,3)、(3,2) —— 都是 $\sum x$

**Hankel 比对称更强**:对称只能配对 (1,3)↔(3,1),而 Hankel 把 (2,2) 也拉进了
同一组。

### 8.4 数一数:6 → 5

| 步骤 | 个数 | 原因 |
|---|---|---|
| 全部元素 | 9 | — |
| 上三角 | 6 | 对称,下三角照抄 |
| 独立数值 | **5** | (1,3) 与 (2,2) 重复,都是 $\sum x^2$ |
| 真正要累加的 | **4** | 其中 $\sum x^0=n$ 是点数,白送 |

所以说"5 个 $\sum x^k$"($k=0,1,2,3,4$)是对的,但要清楚 **6 降到 5 是因为
Hankel 重复,不是因为 $n$ 已知**。

### 8.5 一般公式

| 结论 | 公式 | $m=2$ 验证 |
|---|---|---|
| 矩阵尺寸 | $(m+1)\times(m+1)$ | 3×3 ✓ |
| 上三角元素数 | $\dfrac{(m+1)(m+2)}{2}$ | 6 ✓ |
| **独立数值数** | $\mathbf{2m+1}$ | 5 ✓ |
| 右下角元素 | 恒为 $n$ | ✓ |

> 🔴 **原笔记的错误**:写了"n=4 → 10(4 阶拟合时)"。两处不对:
> 1. **4×4 的矩阵对应 3 次(cubic)拟合**,不是 4 阶。次数 $m$ ↔ 尺寸 $(m+1)^2$。
>    4 阶要 5×5。
> 2. 独立数值**不是 10**,是 $2m+1=7$ 个($\sum x^0$ 到 $\sum x^6$)。
>    $\frac{n(n+1)}{2}$ 是**一般对称矩阵**的公式,套不到 Hankel 上。

✌🏻 只存/只算一半,是处理对称矩阵的常见节省手段。而 Hankel 更省——
$(m+1)^2$ 个元素只需存 $2m+1$ 个数。

---

## 9.  完整例子

$$x=[0,1,2],\qquad y=[1,3,7]$$

### 9.1 构造

$$A=\begin{bmatrix}0&0&1\\1&1&1\\4&2&1\end{bmatrix},
\qquad
A^{\mathsf T}=\begin{bmatrix}0&1&4\\0&1&2\\1&1&1\end{bmatrix}$$

### 9.2 算 $A^{\mathsf T}A$

$$A^{\mathsf T}A=
\begin{bmatrix}0&1&4\\0&1&2\\1&1&1\end{bmatrix}
\begin{bmatrix}0&0&1\\1&1&1\\4&2&1\end{bmatrix}$$

**$A^{\mathsf T}$ 的行**点乘 **$A$ 的列**:

- (1,1):$0\cdot0+1\cdot1+4\cdot4=17$
- (1,2):$0\cdot0+1\cdot1+4\cdot2=9$
- (1,3):$0\cdot1+1\cdot1+4\cdot1=5$

$$A^{\mathsf T}A=\begin{bmatrix}17&9&5\\9&5&3\\5&3&3\end{bmatrix}$$

> 🍏 **术语提醒**:是"**行 × 列**"配对,不是两个矩阵对应位置相乘。
> 后者是 Hadamard 积(见附录 A),完全不同的东西。

✅ **两重校验**:

1. **对称** ✓
2. **Hankel + 公式对照**:$\sum x^4=0+1+16=17$ ✓,$\sum x^3=0+1+8=9$ ✓,
   $\sum x^2=0+1+4=5$ ✓,$\sum x=0+1+2=3$ ✓,$n=3$ ✓。
   反对角线 (1,3)=(2,2)=(3,1)=5 ✓

### 9.3 算 $A^{\mathsf T}y$

$$A^{\mathsf T}y=
\begin{bmatrix}0&1&4\\0&1&2\\1&1&1\end{bmatrix}
\begin{bmatrix}1\\3\\7\end{bmatrix}$$

- $0\cdot1+1\cdot3+4\cdot7=31$
- $0\cdot1+1\cdot3+2\cdot7=17$
- $1\cdot1+1\cdot3+1\cdot7=11$

$$A^{\mathsf T}y=\begin{bmatrix}31\\17\\11\end{bmatrix}$$

### 9.4 解正规方程

$$\begin{bmatrix}17&9&5\\9&5&3\\5&3&3\end{bmatrix}
\begin{bmatrix}c_0\\c_1\\c_2\end{bmatrix}
=\begin{bmatrix}31\\17\\11\end{bmatrix}$$

$$\begin{cases}
17c_0+9c_1+5c_2=31 &(1)\\
9c_0+5c_1+3c_2=17 &(2)\\
5c_0+3c_1+3c_2=11 &(3)
\end{cases}$$

> 🔴 **原笔记这里直接空降答案**,补上消元过程:

$(2)-(3)$ 消去 $c_2$:

$$4c_0+2c_1=6\quad\Longrightarrow\quad 2c_0+c_1=3 \tag{A}$$

$(1)\times3-(3)\times5$ 消去 $c_2$:

$$51c_0+27c_1+15c_2=93$$
$$25c_0+15c_1+15c_2=55$$
$$\Longrightarrow\quad 26c_0+12c_1=38\quad\Longrightarrow\quad 13c_0+6c_1=19 \tag{B}$$

由 (A) 得 $c_1=3-2c_0$,代入 (B):

$$13c_0+6(3-2c_0)=19\ \Longrightarrow\ c_0+18=19\ \Longrightarrow\ c_0=1$$

回代:$c_1=1$;由 (3):$5+3+3c_2=11\Rightarrow c_2=1$。

$$\boxed{c=[1,\ 1,\ 1]^{\mathsf T}\quad\Longrightarrow\quad p(x)=x^2+x+1}$$

**唯一性**:$\det(A^{\mathsf T}A)=17(15-9)-9(27-15)+5(27-25)=102-108+10=4\neq0$,
故解唯一。

### 9.5 ⭕️ 但这题根本不需要正规方程

$x$ 有 3 个互不相同的值,未知数也是 3 个 —— $A$ 是**可逆方阵**,
$\det A=-2\neq0$。直接解 $Ac=y$:

| 代入 | 方程 | 立即得到 |
|---|---|---|
| $x=0$ | $c_2=1$ | $c_2=1$ |
| $x=1$ | $c_0+c_1+c_2=3$ | $c_0+c_1=2$ |
| $x=2$ | $4c_0+2c_1+c_2=7$ | $4c_0+2c_1=6$ |

后两式相减即得 $c_0=1,c_1=1$。**三步出答案**,比先乘出 $A^{\mathsf T}A$ 再消元
干净得多。

左乘 $A^{\mathsf T}$ 在 $A$ 可逆时是**恒等变形**:

$$A^{\mathsf T}Ac=A^{\mathsf T}y\iff c=A^{-1}y$$

答案一样,但绕了一圈,还把数字放大了(17、31 这些数比原始的 0、1、4 大得多)。

✅ **交叉验算**:$\det(A^{\mathsf T}A)=(\det A)^2=(-2)^2=4$,与 9.4 算的一致。

### 9.6 这个例子说明了什么

$A$ 的三列张成整个 $\mathbb{R}^3$,$y$ 本来就落在列空间里,所以投影 = $y$ 自己,
**残差恰好为零**。

$p(x)=x^2+x+1$ 精确穿过 $(0,1)$、$(1,3)$、$(2,7)$ —— 这是**插值**,不是拟合。

⭕️ **最小二乘真正派上用场的是超定情形**($n>m+1$,$A$ 高瘦):无法穿过所有点,
只能求"总体最近"。本例只用来验证公式,不要误以为最小二乘就长这样。

---

## 10.  闭式解与伪逆

正规方程(令导数为零的结果):

$$\boxed{A^{\mathsf T}Ac=A^{\mathsf T}y}$$

当 $A^{\mathsf T}A$ 可逆(即 $A$ 列满秩)时,两边左乘 $(A^{\mathsf T}A)^{-1}$:

$$\boxed{c=(A^{\mathsf T}A)^{-1}A^{\mathsf T}y}$$

⭕️ **这就是最小二乘的闭式解。** 一个公式直接算出来。

🧤 $(A^{\mathsf T}A)^{-1}A^{\mathsf T}$ 有个名字,叫 **Moore–Penrose 伪逆**,
记作 $A^{+}$。所以也写成

$$c=A^{+}y$$

> ⭕️ **前提**:$A^{+}=(A^{\mathsf T}A)^{-1}A^{\mathsf T}$ 这个写法**只在 $A$ 列满秩时成立**。
> 一般情形下 $A^{+}$ 由 SVD 定义,对任意矩阵都存在;此时 $c=A^{+}y$ 给出的是
> **最小范数最小二乘解**(在所有使残差最小的 $c$ 中,长度最短的那个)。

---

## 11.  实践中怎么解

🔴 写成 $(A^{\mathsf T}A)^{-1}$ 是**数学表达**,但**真去求逆矩阵是错误做法**——
既慢又不稳定。实际算法:

| 方法 | 做法 | 特点 |
|---|---|---|
| 正规方程 + Cholesky | 解 $A^{\mathsf T}Ac=A^{\mathsf T}y$ | 最快,但**条件数被平方**,精度差 |
| **QR 分解** | $A=QR$,解 $Rc=Q^{\mathsf T}y$ 回代 | 稳健,通用默认选择 |
| **SVD** | $A=U\Sigma V^{\mathsf T}$,$c=V\Sigma^{+}U^{\mathsf T}y$ | 最稳健,能处理接近奇异的情况 |

### 11.1 为什么"条件数被平方"要命

$$\kappa(A^{\mathsf T}A)=\kappa(A)^2$$

条件数衡量"输入的小扰动会被放大多少倍"。而范德蒙德型矩阵的条件数**本身就随
次数增长得很快**。$\kappa(A)=10^4$ 时,$\kappa(A^{\mathsf T}A)=10^8$ ——
双精度只有约 16 位有效数字,一下就吃掉一半。

QR 和 SVD 直接对 $A$ 分解,**根本不构造 $A^{\mathsf T}A$**,从而绕开这个坑。

### 11.2 NumPy

`np.polyfit` 内部调用 `lstsq`,走的是 **SVD** 路线。它不构造 $A^{\mathsf T}A$,
直接对 $A$ 分解。

> 💡 NumPy 官方推荐用较新的 `numpy.polynomial.Polynomial.fit` 替代旧的
> `np.polyfit`。前者会自动把 $x$ 映射到 $[-1,1]$ 再拟合,条件数好得多
> (就是 7.3 节说的那个技巧)。具体 API 以当前版本文档为准。

### 11.3 那第 7 节的 8 个累加器还能用吗

能,但要分场合:

- ✅ **低次拟合($m\le2$)+ 嵌入式 + $x$ 已归一化**:8 个累加器是最优解,
  内存和算力都省到极致
- ❌ **高次拟合 / 追求精度 / 有现成的线代库**:老老实实用 QR 或 SVD

---

## 12.  一句话串起来

> **理论**:最小二乘 → 求导令零 → 正规方程 $A^{\mathsf T}Ac=A^{\mathsf T}y$
> → 伪逆闭式解 $c=A^{+}y$
>
> **实践**:直接对 $A$ 做 QR 或 SVD,绕开 $A^{\mathsf T}A$

数学上等价,数值上后者稳得多。

---

# 附录 A: 点积

## A.1 定义

**点积**(也叫内积、数量积)是两个**向量**相乘得到一个**标量**的运算。

$$\mathbf a=(a_1,\dots,a_n),\quad \mathbf b=(b_1,\dots,b_n)$$

$$\mathbf a\cdot\mathbf b=a_1b_1+a_2b_2+\cdots+a_nb_n$$

例:$(1,2)\cdot(3,4)=1\times3+2\times4=11$

## A.2 几何形式

$$\mathbf a\cdot\mathbf b=\|\mathbf a\|\,\|\mathbf b\|\cos\theta$$

其中 $\theta$ 是两向量的夹角。所以:

| 点积 | 夹角 | 含义 |
|---|---|---|
| $>0$ | $<90°$ | 方向大致相同 |
| $=0$ | $=90°$ | **两向量垂直** |
| $<0$ | $>90°$ | 方向大致相反 |

⭕️ 常用于判断垂直、求投影,以及物理中的功 $W=\mathbf F\cdot\mathbf s$。

**"$=0$ 即垂直"这条是本笔记的关键**:正规方程的几何推导(第 5 节)靠的就是它。

## A.3 点积 vs 矩阵乘法

- 🟢 **点积**特指两个**向量**相乘得到一个**标量**
- 🟢 $Ac$ 是矩阵乘向量,结果是**向量**,叫**矩阵乘法**
- ⭕️ 矩阵乘法的**每个元素**是一个点积。矩阵乘法 = 点积的"批量打包"

## A.4 两个容易混的地方

| 写法 | 实际运算 | 说明 |
|---|---|---|
| `np.dot(A, c)` | 矩阵乘法 | 名字叫 dot 纯属历史包袱,别被带偏 |
| `A @ c` | 矩阵乘法 | **推荐写法** |
| `A * c` | **逐元素相乘** | Hadamard 积,和点积完全不是一回事 |

## A.5 两种"乘自己"

| 写法 | 尺寸 | 结果 | 名称 |
|---|---|---|---|
| $v^{\mathsf T}v$ | $(1\times n)(n\times1)$ | **标量**,= $\|v\|^2$ | 内积 |
| $vv^{\mathsf T}$ | $(n\times1)(1\times n)$ | **$n\times n$ 矩阵** | 外积 |

顺序不能反。

---

# 附录 B: 规则速查

## B.1 转置

$$(P\pm Q)^{\mathsf T}=P^{\mathsf T}\pm Q^{\mathsf T}$$
$$(PQ)^{\mathsf T}=Q^{\mathsf T}P^{\mathsf T}\qquad\text{(反序)}$$
$$(PQR)^{\mathsf T}=R^{\mathsf T}Q^{\mathsf T}P^{\mathsf T}$$
$$(P^{\mathsf T})^{\mathsf T}=P$$
$$\text{标量的转置等于自身}$$

## B.2 矩阵求导(对列向量 $c$)

$$\frac{\partial}{\partial c}\bigl(c^{\mathsf T}b\bigr)=b$$
$$\frac{\partial}{\partial c}\bigl(b^{\mathsf T}c\bigr)=b$$
$$\frac{\partial}{\partial c}\bigl(c^{\mathsf T}Mc\bigr)=(M+M^{\mathsf T})c=2Mc\quad(M\ \text{对称})$$

## B.3 $A^{\mathsf T}A$ 的性质

| 性质 | 说明 |
|---|---|
| 对称 | $(A^{\mathsf T}A)^{\mathsf T}=A^{\mathsf T}A$ |
| 半正定 | $v^{\mathsf T}A^{\mathsf T}Av=\|Av\|^2\ge0$ |
| 正定 $\iff$ $A$ 列满秩 | 此时可逆,解唯一 |
| $\det(A^{\mathsf T}A)=(\det A)^2$ | 仅 $A$ 为方阵时 |
| $\kappa(A^{\mathsf T}A)=\kappa(A)^2$ | 数值上的代价 |
| $A$ 为范德蒙德型时是 Hankel | $(A^{\mathsf T}A)_{ij}$ 只依赖 $i+j$ |

---