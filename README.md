# myo-tools

### 使用Myo手环实时采集表面肌电信号

使用myo_tools中的save_data文件，可以采集不同手部动作的表面肌电信号。

在使用时，输入手部动作的名称，每种动作的采集时间。

按照打印提示，佩戴Myo手环完成不同手部动作。依次采集每种手部动作，全部采集完毕后，记录为一轮重复。

可以继续采集新一轮重复，也可以保存数据，停止采集。

采集的数据形式为(a,b,c,d)

a是总共采集了多少轮手部动作，b是第几种手部动作，c是采样点个数，d是第几个通道。

### 特别感谢

特别感谢Myo手环社区的Niklas Rosenstein开源了一个Myo-Python库，它可以将Myo手环与电脑连接，库地址https://github.com/NiklasRosenstein/myo-python
