title: php学习笔记--MVC和微框架
date: 2014-10-31 22:43:55
categories: php
tags: [php, mvc]
---

这篇文章是根据[慕课网](http://www.imooc.com/)上的一个视频教程[MVC架构模式分析与设计](http://www.imooc.com/learn/69)总结出来。原视频中讲的非常详细，但学习时间比较长，因此在看视频的时候进行了总结，方便日后温故而知新。


## 基本MVC

> 
- **Model模型：**模型的工作是按要求从数据库、接口取出数据（不全面的一个解释）
- **View视图：**能只管看到的Web界面
- **Controller控制器：**向系统发出指令的工具和帮手


**MVC工作流程：**

1. 浏览者 -> 调用控制器，对他发出指令
2. 控制器 -> 按指令选取一个合适的模型
3. 模型 -> 按控制器指令取相应数据
4. 控制器 -> 按指令选取相应视图
5. 视图 -> 把第三步取到的数据按用户想要的样子显示处理

**简单MVC实例：**

控制器`testController.class.php`
```
<?php
    class testController {
        function show() {
            $testModel = new testModel();
            $testModel->get();
            
            $testView = new testView();
            $testView->display();
        }
    }
?>
```

模型`testModel.class.php`
```
<?php
    class testModel {
        function get() {
            //获取数据
            //...
            return "hello world";
        }
    }
?>
```

视图`testView.class.php`
```
<?php
    class testView {
        function display($data) {
            //组织美化数据
            echo $data;
        }
    }
?>
```

`test.php`
```
<?php
    require_once('testController.class.php);
    require_once('testModel.class.php);
    require_once('testView.class.php);
    
    $testController = new testController();
    $testController->show();
    
?>
```


## 系统目录规范

对上面的文件规范化，按如下目录组织：
![](/img/mvc_base.png)

对控制器、模型、视图的操作进行封装`function.php`
```
<?php
    //控制器简易调用函数
    function C($name, $method) {
        require_once('libs/Controller/'.$name.'Controller.class.php');
        
        $controller = $name.'Controler';
        $obj = new $controller();
        $obj->method();
    }
    
    //模型简易调用函数
    function M($name) {
        require_once('libs/Model/'$name.'Model.class.php');
        $model = $name.'Model';
        $obj = new $model();
        return $obj;
    }
    
    //视图简易调用函数
    function V($name) {
        require_once('libs/View/'.$name.'View.class.php');
        $view = $naem.'View';
        $obj = new $view();
        return $obj;
    }
    
    //返回安全的字符串（不安全字符进行转义）
    function daddslashes($str) {
        return (!get_magic_quotes_gpc()) ? addslashes($str) : $str;
    }
?>
```

按照这种方式组织的入口文件`index.php`：
```
<?php
    //url形式 index.php?controller=控制器名&method=方法名
    require_once('function.php');
    $controllerAllow = array('test', 'index');
    $methodAllow = array('test', 'index', 'show');
    
    $controller = in_array($_GET['controller'], $controllerAllow) ? daddslashes(_GET['controller']) ： 'index';
    $method = in_array($_GET['method'], $methodAllow) ? daddslashes($_GET['method']) : 'index';
    C($controller, $method);
?>
```

按照这种方式组织的控制器`testController.class.php`：
```
<?php
    class testController {
        function show() {
            $testModel = M('test');
            $data = $testModel->get();
            $testView = V('test');
            $testView->display($data);
        }
    }
?>
```

## PHP操作Mysql类的封装

**封装目的：**
1. 减少代码冗余，提高开发速度
2. 降低编程错误
3. 便于维护升级


**实例：`mysql.class.php`**
```
<?php
    class mysql {
        /**
        * 错误输出
        **/
        function err($error) {
            die("对不起，您的操作有误，错误原因为：$error");
        }
        
        /**
        * 连接数据库
        *
        * @param string $config 配置数组 array($dbhost $dbuser $dbpsw $dbname $dbcharset)
        * @return bool 连接成功或不成功
        **/
        function connect($config) {
            extract($config);
            if(!($con = mysql_connect($dbhost, $dbuser, $dbpsw))) {//mysql_connect连接数据库函数
                $this->err(mysql_error());
            }
            if(!(mysql_select_db($dbname, $con)) {//mysql_select_db选择库的函数
                $this->err(mysql_error());
            }
            mysql_query("set names ".$dbcharset);//使用mysql_query设置编码  格式:mysql_query("set names utf8")
        }
        
        /**
        *执行sql语句
        *
        * @param string $sql
        * @return bool 返回执行成功、资源或执行失败
        **/
        function query($sql) {
            if(!(query = mysql_query($sql))) {//使用mysql_query函数执行sql语句
                $this->err($sql."<br/>".mysql_error()); //mysql_error 报错
            } else {
                return $query;
            }
        }
        
        /**
        * 列表
        *
        * @param source $query sql语句通过mysql_query执行出来的资源
        * @return array 返回列表数组
        **/
        function findAll($query) {
            while($rs=mysql_fetch_array($query, MYSQL_ASSOC)) {//mysql_fetch_array函数把资源转换为数组，一次转换出一行出来
                $list[] = $rs;        
            }
            
            return isset($list) ? $list : "";
        }
        
        /**
        * 单条
        *
        * @param source $query sql语句通过mysql_query执行出来的资源
        * @return array 返回单条信息数组
        **/
        function findOne($query) {
            $rs = mysql_fetch_array($query, MYSQL_ASSOC);
            return $rs;
        }
        
        /**
        * 指定行的指定字段的值
        *
        * @param source $query sql语句通过mysql_query执行出来的资源
        * @return array 返回指定行的指定字段的值
        **/
        function findResult($query, $row = 0, $field = 0) {
            $rs = mysql_result($query, $row, $field);
            return $rs;
        }
        
        /**
        * 添加函数
        *
        * @param string $table 表名
        * @param array $arr 添加数组（包含字段和值的一维数组）
        **/
        function insert($table, $arr) {
            //$sql = "insert into 表名(多个字段) values(多个值)";
            foreach($arr as $key => $value) {//foreach循环数组
                $value = mysql_real_escape_string($value);
                $keyArr[] = "`".$key."`";//把$arr数组中的键名保存到$keyArr数组当中
                $valueArr[] = "'".$value."'";//把$arr数组当中的键值保存到$valueArr当中，因为值多为字符串，而sql语句里面insert当中如果值是字符串的话需要加单引号
            }
            
            $keys = implode(",", $keyArr);//implode函数是把数组组合成字符串 implode(分隔符, 数组)
            $values = implode(",", $valueArr);
            $sql = "insert into ".$table."(".$keys.") values(".$values.")";
            $this->query($sql); //调用类自身的query(执行)方法执行这条sql语句
            return mysql_insert_id();
        }
        
        /**
        * 修改函数
        *
        * @param string $table 表名
        * @param array $arr 修改数组（包含字段和值的一维数组）
        * @param string $where  条件
        **/
        function update($table, $arr, $where) {
            //update 表名 set 字段=字段值 where...
            foreach($arr as $key => $value) {
                $value = mysql_real_escape_string($value);
                $keyAndvalueArr[] = "`".$key."`"="'".$value."'";
            }
            $keyAndvalues = implode(",", $keyAndvalueArr);
            $sql = "update ".$table." set ".$keyAndvalues." where ".$where;
            $this->query($sql);
        }
        
        /**
        * 删除函数
        *
        * @param string $table 表名
        * @param string $where  条件
        **/
        function del($table, $where) {
            $sql = "delete form ".$table." where ".$where;
            $this->query($sql);
        }
    }
?>
```



## 简单工厂模式

**工厂模式是做什么的：**
例如现在有mysql操作类和mssql操作类
```
class factory {
    static function creat($type) {
        return new $type;
    }
}
$obj = factory::creat('mysql');
```

**好处：**统一管理对象的实例化，便于扩展维护


## DB工厂类
**实例：`DB.class.php`**
```
<?php
class DB { //类名在php里面是一个全局变量
    public static $db;
    
    public static function init($dbtype, $config) {
        self::$db = new $dbtype;
        self::$db->connect($config);
    }
    
    public static function query($sql) {
        return self::$db->query($sql);
    }
    
    public static function findALL($sql) {
        $query = self::$db->query($sql);
        return self::$db->findAll($query);
    }
    
    public static function findOne($sql) {
        $query = self::$db->query($sql);
        return self::$db->findOne($query);
    }
    
    public static function findResult($sql, $row=0, $field=0) {
        $query = self::$db->query($sql);
        return self::$db->findResult($query, $row, $field);
    }
    
    public static function insert($table, $arr) {
        return self::$db->insert($table, $arr);
    }
    
    public static function update($table, $arr, $where) {
        return self::$db->update($table, $arr, $where);
    }
    
    publict static function del($table, $where) {
        return self::$db->del($table, $where);
    }
}

?>
```


## 视图引擎工厂类

**实例：`VIEW.class.php`**
```
<?php
class VIEW {
    public static $view;
    
    public static function init($viewtype, $config) {
        self::$view = new $viewtype;
        
        /*例如Smarty模板引擎进行一下配置：
        $smarty = new Smarty();
        $smarty->left_delimiter = $config["left_delimiter"];//左定界符
        $smarty->right_delimiter = $config["right_delimiter"];//右定界符
        $smarty->template_dir = $config["template_dir"];//html模板的地址
        $smarty->compile_dir = $config["compile_dir"];//模板编译生成的文件
        $smarty->cache_dir = $config["cache_dir"];//缓存
        */
        foreach($config as $key => $value){
            self::$view->$key = $value;
        }
    }
    
    public static function assign($data) {
        foreach($data as $key => $value) {
            self::$view->assign($key, $value);
        }
    }
    
    public static function display($template) {
        self::$view->display($template)
    }
}

?>

```


## 微型框架编写
**框架组织结构：**
1. 函数库：一些零散的函数，放在一个文件里面（如：`function.php`）
2. 类库
    - 视图引擎库（如：Smarty类库）
    - DB引擎库（如：mysql.class.php）
    - 核心库，如上面两个工厂类
3. require清单文件
4. 启动引擎程序：完成一些列的初始化，调用控制器

**目录实例：**
![](/img/mini_framework.png)



**require清单文件`include.list.php`：**
```
<?php
    $paths = array(
        'function/function.php',
        'libs/core/DB.class.php',
        'libs/core/VIEW.class.php',
        'libs/db/mysql.class.php',
        'libs/views/Smarty/Smarty.class.php'
    );
?>
```


**启动引擎`GSS.php`：**
```
<?php
    $currentdir = dirname(__FILE__);
    include_once($currentdir.'inlude.list.php');
    foreach($paths as $path) {
        include_once($currentdir.'/'.$path);
    }
    
    class GSS {
        public static $controller;
        public static $method;
        public static $config;
        
        private static function init_db() {
            DB::init('mysql', self::$config['dbconfig']);
        }
        private static function init_view() {
            VIEW::init('Smarty', self::$confi['viewconfig']);
        }
        private static init_controller() {
            self::$controller = isset($_GET['controller']) ? daddslashes($_GET['controller']) : 'index';
        }
        private static init_method() {
            self::$method = isset($_GET['method']) ? daddslashes($_GET['method']) : 'index'; 
        }
        public static function run($config) {
            self::$config = $config;
            self::init_db();
            self::init_view();
            self::init_controller();
            self::init_method();
            C(self::$controller, self::$method);
        }
    }
?>

```


## 入口文件和配置文件

**入口文件：`index.php`**

```
<?php
    header("Content-type; text/html; charset=utd-8");
    session_start();
    require_once('config.php');
    require_once('framework/GSS.php');
    PC::run($config);
?>
```


**配置文件：`config.php`**
```
<?php
    $config = array(
        'viewconfig' => array(
            'left_delimiter' => '{',
            'right_delimiter' => '}',
            'template_dir' => 'tpl',
            'compile_dir' => 'data/template_C'
        ),
        'dbconfig' => array(
            'dbhost' => 'localhsot',
            'dbuser' => 'root',
            'dbpsw' => '123',
            'dbname' => 'newsreport',
            'dbcharset' => 'utf8'
        )
    );
?>
```