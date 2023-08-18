# elasticsearch-8.9采坑

一、操作步骤

​		我采用的是非docker版本，直接使用elasticsearch-8.9.0-linux-x86_64.tar.gz 解压进行直接启动操作。

​		步骤1：从[csdn网站](https://download.csdn.net/download/h735004790/88152865)可以下载elasticsearch-8.9.0-linux-x86_64.tar.gz；

​		步骤2：解压.tar.gz文件，获得

```
tar -xzvf elasticsearch-8.9.0-linux-x86_64.tar.gz
```

![image-20230818152834036](./elasticsearch.assets/es操作.png)

​		步骤3：修改配置文件，需要修改的文件包括两个elasticsearch.yml和jvm.options.d

![image-20230818153213137](./elasticsearch.assets/配置文件修改.png)

​		(1)修改elasticsearch.yml 

```shell

xpack.security.enabled: false
xpack.security.enrollment.enabled: false
xpack.security.http.ssl:
  enabled: false
  keystore.path: certs/http.p12
xpack.security.transport.ssl:
  enabled: true
  verification_mode: certificate
  keystore.path: certs/transport.p12
  truststore.path: certs/transport.p12
cluster.initial_master_nodes: ["mk"]
http.host: 0.0.0.0

```

​		(2)修改jvm.options

```
-Xms1g
-Xmx1g
```

​		步骤4：安装ik分词器这里直接使用elasticsearch-analysis-ik-8.9.0.zip，将它解压至elasticsearch-8.9.0/plugins/ik，注意检查文件不要有遗漏。[分词器安装](https://blog.csdn.net/qq_39939541/article/details/131619209)

​		步骤5：启动服务，直接执行no_docker_run.sh，对应脚本的命令如下：

```
./elasticsearch-8.9.0/bin/elasticsearch -d
```

​		步骤6：验证服务是否真正启动，可以使用curl命令或者使用网页打开，注意端口和ip不要写错。

![image-20230818154035739](./elasticsearch.assets/验证es.png)

二、常见问题

（1）elasticsearch.BadRequestError: BadRequestError(400, 'illegal_argument_exception', 'Custom Analyzer [ik_analyzer] failed to find tokenizer under name [ik_max_word]')这个错误提示的是在使用 Elasticsearch 的时候，尝试使用一个自定义分析器 `ik_analyzer`，但是在配置中没有找到名为 `ik_max_word` 的分词器。`ik_max_word` 是 IK 分词器的一个模式。IK 分词器是一个中文分词器，通常用于 Elasticsearch 中处理中文文本。很大可能没有安装指定的ik分词器。

（2）Native controller process has stopped - no new native processes can be start，本问题的[解决方法](https://www.jianshu.com/p/376042ce0faf)

