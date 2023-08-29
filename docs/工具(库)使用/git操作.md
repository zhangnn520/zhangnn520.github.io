#                                               git基操

## 一、 [如何将本地整个文件夹上传至GitHub](https://www.cnblogs.com/wwwzgy/p/15056431.html)

​		Github是一个可以进行代码共享、共同开发、程序版本管理的平台。没有Github帐号的程序员肯定不是一个合格的程序员，有了Github帐号还得会用它。今天先来讲入门操作：如何把编写好的程序一次性全部上传到Github，或者说如何把一个文件及其子文件夹里所有的文件上传到Github中。操作步骤如下：

1. 在Github上创建一个空的Repository，可以是一个private的，也可以是一个public的。我这里建一个private的repository为例：创建立一个名叫security的repository。
2. 下载安装Git工具,下载地址https://git-scm.com/downloads，安装好后，右键菜单上会增加Git GUI here和Git Bash here两个选项。
3. 确认已安装微软.net 4.7 以上版本的framework。
4. 打开想要上传的文件夹，点右键，选择Git Bash here，弹出Git命令框。
5. 输入命令git init，这时在文件夹中会多出一个.git的文件夹
6. 输入命令git remote add origin 【命令用于将一个远程仓库添加到当前的本地仓库中，并给它取一个名字作为别名（通常是 `origin`）】，例如，git remote add origin https://github.com/wwzgy/security.git。输完后，可用命令git remote -v查看是否输得准确，如果有错误，可以输入git remote rm origin，然后再重新输git remote add origin 【你的github仓库地址】。
7. 如果已有不需要进行创建，输入命令 ssh-keygen，这时会在系统盘下你的用户名下的.ssh文件夹里生成一个id_rsa.pub和一个id_rsa文件，用记事本打开其中的id_rsa.pub，复制所有字符，然后登录github中你的帐户，在帐户setting中选择SSH and GPG Key，创建一个新的SSH Key，把刚才复制的字符全部粘贴到这里。
8. 继续在git命令框里输入git add . （注意add后面有一个空格和一个点，意思是将本文件夹内所有文件添加到本地缓存区）。
9. 输入命令git commit -m "first commit"
10. 输入命令git push origin HEAD:master，如果弹出对话框则按要求输入用户名和密码，如果此命令报错，尝试后面两步后再次执行push的命令。
11. git pull --rebase origin master （将两个分支合并）*
12. git config http.sslVerify "false" （不做ssl验证）
13. 如果文件夹中有target文件夹，默认是不上传这个文件夹下的文件的，如果要上传，需要用命令git add -f target将其加入到本地缓存区，然后再进行上面的commit和push。
14. 如果需要设定用户名，用命令git config user.name "你的用户名"；如果要设定密码，用命令git config user.password "你的密码"。





参考链接：

1、https://www.cnblogs.com/wwwzgy/p/15056431.html
