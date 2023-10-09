# stock_prediction

这是本项目的解决方案简介。首先我们将介绍测试相关代码，然后介绍训练相关代码。python环境为3.9，torch版本为1.13.0
我们在算法挑战赛“基于财报的次日价格涨跌预测挑战赛”中获得了第一名。由于本比赛队员并不全部来自于北京大学，因此未更改队名。
比赛相关情况可以从https://challenge.xfyun.cn/topic/info?type=next-day-price-fluctuations
获取
![image](https://github.com/xuxiran/stock_prediction/assets/48015859/62072159-6b88-4063-87e0-7745a58673e3)

注意：我们重新对数据进行了标注，并形成了数据集Financial_Report_202306_202309。需要联系我们请求数据集，您可以联系2001111407@stu.pku.edu.cn。
由于可能存在的法律风险，我们只接受中国大陆国内edu邮箱的请求，并且请求者需要实名承诺只用于学术研究。


解决方案
团队成员认为，当前主办方提供的数据不足以成功训练一个财报-涨跌模型。因此，团队从公开网站上分工收集不同来源的财报。
经过数据清洗后，我们共整理出62284条财报pdf的文本数据（使用PyPDF2），并对其进行了标签标注。
这些文本数据及其标签将被称为Financial_Report_202306_202309。将其作为我们的训练集。
我们将主办方提供的7374条数据作为验证集。我们将pdf读取出的文本作为输入，使用次日股票数据作为输出。使用Bert-chinese作为预训练模型。
最终，我们在主办方提供的测试集（未知）中取得了0.81的f1分数（第一名）。

Financial_Report_202306_202309数据集
Financial_Report_202306_202309数据集被保存在train_data.pkl中。已经在保存在文件中


测试相关代码：您只需要运行test.py即可
我们假设'project/xfdata' 下存在495份pdf文件，对应于赛时文件。
如果文件路径错误，会导致无法推理。请手动修改文件路径，或者也可以联系我重新提交代码。
test.py运行后，您将在'project/prediction_result/'下一份result.csv。
如果test_labels和比赛时一样，您可以获得0.81的F1分数


训练相关代码：您只需要运行train.py即可
我们使用了bert-base-chinese训练模型，下载地址为https://huggingface.co/bert-base-chinese
具体分析可见技术报告或代码

额外分析相关代码：
由于数据集较大，可能包含了测试集内容。因此，我们在本地模仿了一个相同大小的测试集(K495)。
我们将训练集中包括K495字符串的字符串删除。删除后，还剩下61853条数据。接着我们使用剩下的数据训练，K495测试.
我们仍然获得了类似的结果。
如果希望完成技术报告中的数据规模分析，请在train.py中训练数据后，加上两行代码如下：
train_data = train_data[:10000]
train_labels = train_labels[:10000]
针对数据规模分析的不同条件，请将10000修改成20000-60000的不同范围。

写在最后：由于团队暂时没有法律顾问，我们对PKU_financial_report数据集的任何使用、传播呈谨慎态度。
并且PKU_financial_report数据集虽来自公开数据，但经搜集整理后，可被视为本团队保密信息。
因此本数据集不允许任何团体、机构或个人用于任何商业用途。
