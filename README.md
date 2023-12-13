# fault_predict
1、项目描述 <br>
&nbsp;&nbsp;&nbsp;&nbsp;根据机器故障的振动幅度历史数据，预测机器在未来某个时点的振幅

2、实现思路 <br>
&nbsp;&nbsp;&nbsp;&nbsp;代码和模型，有pycharm和jupyter两个版本的代码，jupyter代码可看到训练过程，以及用测试数据验证的效果。

&nbsp;&nbsp;&nbsp;&nbsp;代码逻辑是将原始数据集拆分为训练集和测试集，每个集合样本是由原来的数据集派生出来，派生思路是当前第N个时刻，前N-10时刻的数据作为样本，而N时刻的数据作为标签，而具体是用前10个数据，还是前100个，是可配的。

&nbsp;&nbsp;&nbsp;&nbsp;而在训练或测试时，并没有使用全量样本数据，而是随机从全量数据挑选出一定数量样本，这个具体数目也是可配置的，目前基线版本使用的是训练集2000个样本，测试集200个。

&nbsp;&nbsp;&nbsp;&nbsp;模型采用GRU+注意力机制。

3、项目运行效果 <br>
&nbsp;&nbsp;&nbsp;&nbsp;下图红色折色折线为原始数据，蓝色折线为相应的预测数据，拟合效果见图。
<img width="524" alt="66a3970c4229bb904bb3f680abf9557" src="https://github.com/tgltt/fault_predict/assets/36066270/3053f335-4fdc-40f5-ba66-9bef6fc77842">




