# 🚀 HADOOP SINGLE-NODE CLUSTER - COMPLETE SETUP GUIDE

**VPS Specs**: 16GB RAM | 200GB HDD | 4 vCPUs  
**Target**: Production-ready single-node Hadoop cluster với SSH access từ local machine

---

## 📋 **TABLE OF CONTENTS**

1. [System Overview](#system-overview)
2. [Initial VPS Setup](#initial-vps-setup)
3. [Java Installation](#java-installation)
4. [User & SSH Configuration](#user--ssh-configuration)
5. [Hadoop Installation](#hadoop-installation)
6. [Hadoop Configuration](#hadoop-configuration)
7. [HDFS Setup & Formatting](#hdfs-setup--formatting)
8. [Starting Services](#starting-services)
9. [Verification & Testing](#verification--testing)
10. [File Transfer Setup (SCP)](#file-transfer-setup-scp)
11. [Common Operations](#common-operations)
12. [Performance Optimization](#performance-optimization)
13. [Troubleshooting](#troubleshooting)
14. [Maintenance & Monitoring](#maintenance--monitoring)

---

## 🖥️ **SYSTEM OVERVIEW**

### **Architecture**:

```
┌─────────────────────────────────────┐
│           Single VPS Node           │
│  ┌─────────────────────────────────┐│
│  │        NameNode                 ││
│  │        DataNode                 ││
│  │        ResourceManager          ││
│  │        NodeManager              ││
│  │        JobHistoryServer         ││
│  │        SecondaryNameNode        ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
```

### **Resource Allocation**:

- **HDFS**: 150GB storage space
- **YARN**: 12GB RAM (4GB reserved for OS)
- **JVM Heap**: 8GB total across services
- **CPU**: All 4 cores available

---

## 🔧 **INITIAL VPS SETUP**

### **Step 1: Update System**

```bash
# Connect to VPS
ssh root@your-vps-ip

# Update package lists
apt update && apt upgrade -y

# Install essential packages
apt install -y curl wget vim htop unzip software-properties-common
```

### **Step 2: Set Hostname & Timezone**

```bash
# Set hostname
hostnamectl set-hostname hadoop-master
echo "127.0.0.1 hadoop-master" >> /etc/hosts

# Set timezone (adjust as needed)
timedatectl set-timezone Asia/Ho_Chi_Minh

# Verify
hostnamectl
timedatectl
```

### **Step 3: Configure Firewall**

```bash
# Install UFW if not present
apt install -y ufw

# Allow SSH
ufw allow 22

# Allow Hadoop web interfaces (optional, for monitoring)
ufw allow 9870  # NameNode Web UI
ufw allow 8088  # ResourceManager Web UI
ufw allow 19888 # JobHistory Web UI

# Enable firewall
ufw --force enable
ufw status
```

---

## ☕ **JAVA INSTALLATION**

### **Install OpenJDK 11**

```bash
# Install Java 11
apt install -y openjdk-11-jdk

# Verify installation
java -version
javac -version

# Set JAVA_HOME
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> /etc/environment
echo 'export PATH=$PATH:$JAVA_HOME/bin' >> /etc/environment
source /etc/environment

# Verify JAVA_HOME
echo $JAVA_HOME
```

---

## 👤 **USER & SSH CONFIGURATION**

### **Step 1: Create Hadoop User**

```bash
# Create dedicated hadoop user
adduser hadoop
usermod -aG sudo hadoop

# Switch to hadoop user
su - hadoop
```

### **Step 2: SSH Key Setup (Passwordless SSH)**

```bash
# Generate SSH key pair (as hadoop user)
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# Add public key to authorized_keys
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh

# Test passwordless SSH to localhost
ssh localhost
exit
```

### **Step 3: Setup SSH Access from Local Machine**

```bash
# On your local machine, copy your public key to VPS
ssh-copy-id hadoop@your-vps-ip

# Test connection from local machine
ssh hadoop@your-vps-ip
```

### **Step 4: Configure SSH for Convenience**

```bash
# On local machine, create SSH config
cat > ~/.ssh/config << 'EOF'
Host hadoop-cluster
    HostName your-vps-ip
    User hadoop
    Port 22
    IdentityFile ~/.ssh/id_rsa
EOF
```

---

## 📦 **HADOOP INSTALLATION**

### **Step 1: Download & Extract Hadoop**

```bash
# As hadoop user, go to home directory
cd /home/hadoop

# Download Hadoop 3.3.6 (latest stable)
wget https://archive.apache.org/dist/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz

# Extract
tar -xzf hadoop-3.3.6.tar.gz
mv hadoop-3.3.6 hadoop
rm hadoop-3.3.6.tar.gz
```

### **Step 2: Set Environment Variables**

```bash
# Add environment variables to ~/.bashrc
cat >> ~/.bashrc << 'EOF'

# Hadoop Environment Variables
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export HADOOP_HOME=/home/hadoop/hadoop
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME
EOF

# Apply changes
source ~/.bashrc

# Verify
hadoop version
```

### **Step 3: Set JAVA_HOME in Hadoop Config**

```bash
# Set JAVA_HOME in hadoop-env.sh
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
```

---

## ⚙️ **HADOOP CONFIGURATION**

### **Step 1: Core Configuration (core-site.xml)**

```bash
# Create core-site.xml configuration
cat > $HADOOP_HOME/etc/hadoop/core-site.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <!-- Default filesystem URI -->
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://hadoop-master:9000</value>
        <description>NameNode URI</description>
    </property>

    <!-- Temporary directory -->
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/home/hadoop/hadoop/tmp</value>
        <description>Temporary directory for Hadoop</description>
    </property>

    <!-- Proxy user for web interfaces -->
    <property>
        <name>hadoop.proxyuser.hadoop.groups</name>
        <value>*</value>
    </property>
    <property>
        <name>hadoop.proxyuser.hadoop.hosts</name>
        <value>*</value>
    </property>
</configuration>
EOF
```

### **Step 2: HDFS Configuration (hdfs-site.xml)**

```bash
# Create hdfs-site.xml configuration
cat > $HADOOP_HOME/etc/hadoop/hdfs-site.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <!-- Replication factor for single node -->
    <property>
        <name>dfs.replication</name>
        <value>1</value>
        <description>Default block replication</description>
    </property>

    <!-- NameNode data directory -->
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>/home/hadoop/hadoop/data/namenode</value>
        <description>Directory for storing namespace and transaction logs</description>
    </property>

    <!-- DataNode data directory -->
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>/home/hadoop/hadoop/data/datanode</value>
        <description>Directory for storing blocks</description>
    </property>

    <!-- NameNode web interface -->
    <property>
        <name>dfs.namenode.http-address</name>
        <value>hadoop-master:9870</value>
    </property>

    <!-- Block size (128MB) -->
    <property>
        <name>dfs.blocksize</name>
        <value>134217728</value>
    </property>

    <!-- Permissions -->
    <property>
        <name>dfs.permissions.enabled</name>
        <value>false</value>
    </property>
</configuration>
EOF
```

### **Step 3: YARN Configuration (yarn-site.xml)**

```bash
# Create yarn-site.xml configuration
cat > $HADOOP_HOME/etc/hadoop/yarn-site.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <!-- ResourceManager hostname -->
    <property>
        <name>yarn.resourcemanager.hostname</name>
        <value>hadoop-master</value>
    </property>

    <!-- ResourceManager web interface -->
    <property>
        <name>yarn.resourcemanager.webapp.address</name>
        <value>hadoop-master:8088</value>
    </property>

    <!-- NodeManager services -->
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>

    <!-- Memory allocation (12GB out of 16GB) -->
    <property>
        <name>yarn.nodemanager.resource.memory-mb</name>
        <value>12288</value>
    </property>

    <!-- CPU cores -->
    <property>
        <name>yarn.nodemanager.resource.cpu-vcores</name>
        <value>4</value>
    </property>

    <!-- Application memory -->
    <property>
        <name>yarn.app.mapreduce.am.resource.mb</name>
        <value>2048</value>
    </property>

    <!-- Container memory limits -->
    <property>
        <name>yarn.scheduler.minimum-allocation-mb</name>
        <value>512</value>
    </property>
    <property>
        <name>yarn.scheduler.maximum-allocation-mb</name>
        <value>12288</value>
    </property>
</configuration>
EOF
```

### **Step 4: MapReduce Configuration (mapred-site.xml)**

```bash
# Create mapred-site.xml configuration
cat > $HADOOP_HOME/etc/hadoop/mapred-site.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <!-- MapReduce framework -->
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>

    <!-- JobHistory Server -->
    <property>
        <name>mapreduce.jobhistory.address</name>
        <value>hadoop-master:10020</value>
    </property>
    <property>
        <name>mapreduce.jobhistory.webapp.address</name>
        <value>hadoop-master:19888</value>
    </property>

    <!-- Map task memory -->
    <property>
        <name>mapreduce.map.memory.mb</name>
        <value>2048</value>
    </property>

    <!-- Reduce task memory -->
    <property>
        <name>mapreduce.reduce.memory.mb</name>
        <value>2048</value>
    </property>

    <!-- JVM heap size -->
    <property>
        <name>mapreduce.map.java.opts</name>
        <value>-Xmx1640m</value>
    </property>
    <property>
        <name>mapreduce.reduce.java.opts</name>
        <value>-Xmx1640m</value>
    </property>
</configuration>
EOF
```

---

## 💾 **HDFS SETUP & FORMATTING**

### **Step 1: Create Required Directories**

```bash
# Create data directories
mkdir -p /home/hadoop/hadoop/data/namenode
mkdir -p /home/hadoop/hadoop/data/datanode
mkdir -p /home/hadoop/hadoop/tmp

# Set permissions
chmod 755 /home/hadoop/hadoop/data/namenode
chmod 755 /home/hadoop/hadoop/data/datanode
chmod 755 /home/hadoop/hadoop/tmp
```

### **Step 2: Format NameNode**

```bash
# Format HDFS (ONLY do this once!)
$HADOOP_HOME/bin/hdfs namenode -format -force

# Look for "Storage directory has been successfully formatted" message
```

---

## 🚀 **STARTING SERVICES**

### **Step 1: Start HDFS Services**

```bash
# Start NameNode and DataNode
$HADOOP_HOME/sbin/start-dfs.sh

# Verify HDFS services
jps
# Should see: NameNode, DataNode, SecondaryNameNode
```

### **Step 2: Start YARN Services**

```bash
# Start ResourceManager and NodeManager
$HADOOP_HOME/sbin/start-yarn.sh

# Verify YARN services
jps
# Should see: ResourceManager, NodeManager (+ HDFS services)
```

### **Step 3: Start JobHistory Server**

```bash
# Start JobHistory Server
$HADOOP_HOME/bin/mapred --daemon start historyserver

# Final verification
jps
# Should see all 6 services:
# - NameNode
# - DataNode
# - SecondaryNameNode
# - ResourceManager
# - NodeManager
# - JobHistoryServer
```

---

## ✅ **VERIFICATION & TESTING**

### **Step 1: Web Interface Access**

Open these URLs in your browser (replace `your-vps-ip`):

- **NameNode Web UI**: http://your-vps-ip:9870
- **ResourceManager Web UI**: http://your-vps-ip:8088
- **JobHistory Web UI**: http://your-vps-ip:19888

### **Step 2: HDFS Health Check**

```bash
# Check HDFS status
hdfs dfsadmin -report

# Expected output should show:
# - Configured Capacity > 0
# - Live datanodes (1)
# - No missing blocks
```

### **Step 3: YARN Health Check**

```bash
# Check YARN nodes
yarn node -list

# Expected output:
# - Total Nodes: 1
# - Node State: RUNNING
```

### **Step 4: Test HDFS Operations**

```bash
# Create test directory
hdfs dfs -mkdir /test

# Upload a test file
echo "Hello Hadoop!" > test.txt
hdfs dfs -put test.txt /test/

# List files
hdfs dfs -ls /test/

# Download file
hdfs dfs -get /test/test.txt downloaded.txt
cat downloaded.txt

# Clean up
hdfs dfs -rm /test/test.txt
hdfs dfs -rmdir /test
rm test.txt downloaded.txt
```

### **Step 5: Test MapReduce Job**

```bash
# Run Pi calculation example
yarn jar $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.3.6.jar pi 2 10

# Should complete successfully with Pi approximation result
```

---

## 📁 **FILE TRANSFER SETUP (SCP)**

### **From Local Machine to Hadoop Cluster:**

```powershell
# Upload single file (PowerShell)
scp -i "C:\Users\tuan\.ssh\hadoop_key" local-file.txt hadoop@167.71.203.123:/home/hadoop/

# Upload directory
scp -i "C:\Users\tuan\.ssh\hadoop_key" -r local-directory\ hadoop@167.71.203.123:/home/hadoop/

# Using SSH config alias (more convenient)
scp local-file.txt hadoop-cluster:/home/hadoop/
```

### **From Hadoop Cluster to Local Machine:**

```powershell
# Download single file
scp -i "C:\Users\tuan\.ssh\hadoop_key" hadoop@167.71.203.123:/home/hadoop/remote-file.txt ./

# Download directory
scp -i "C:\Users\tuan\.ssh\hadoop_key" -r hadoop@167.71.203.123:/home/hadoop/remote-directory/ ./

# Download from HDFS
ssh -i "C:\Users\tuan\.ssh\hadoop_key" hadoop@167.71.203.123 "hdfs dfs -get /hdfs-path/file.txt /tmp/"
scp -i "C:\Users\tuan\.ssh\hadoop_key" hadoop@167.71.203.123:/tmp/file.txt ./
```

### **Batch Transfer Script (PowerShell)**

```powershell
# Create transfer script on local machine
@'
# Hadoop File Transfer Script (PowerShell)

$VPS_IP = "167.71.203.123"
$SSH_KEY = "C:\Users\tuan\.ssh\hadoop_key"
$HADOOP_USER = "hadoop"

function Upload-ToHDFS {
    param(
        [string]$LocalPath,
        [string]$HDFSPath
    )

    Write-Host "Uploading $LocalPath to HDFS:$HDFSPath"
    scp -i $SSH_KEY $LocalPath "${HADOOP_USER}@${VPS_IP}:/tmp/"
    $filename = Split-Path $LocalPath -Leaf
    ssh -i $SSH_KEY "${HADOOP_USER}@${VPS_IP}" "hdfs dfs -put /tmp/$filename $HDFSPath"
    ssh -i $SSH_KEY "${HADOOP_USER}@${VPS_IP}" "rm /tmp/$filename"
}

function Download-FromHDFS {
    param(
        [string]$HDFSPath,
        [string]$LocalPath
    )

    Write-Host "Downloading HDFS:$HDFSPath to $LocalPath"
    $filename = Split-Path $HDFSPath -Leaf
    ssh -i $SSH_KEY "${HADOOP_USER}@${VPS_IP}" "hdfs dfs -get $HDFSPath /tmp/"
    scp -i $SSH_KEY "${HADOOP_USER}@${VPS_IP}:/tmp/$filename" $LocalPath
    ssh -i $SSH_KEY "${HADOOP_USER}@${VPS_IP}" "rm /tmp/$filename"
}

# Usage examples:
# Upload-ToHDFS "data.csv" "/input/data.csv"
# Download-FromHDFS "/output/results.txt" "./results.txt"
'@ | Out-File -FilePath "hadoop-transfer.ps1" -Encoding UTF8
```

---

## 🔧 **COMMON OPERATIONS**

### **Start/Stop Cluster**

```bash
# Stop everything
$HADOOP_HOME/bin/mapred --daemon stop historyserver
$HADOOP_HOME/sbin/stop-yarn.sh
$HADOOP_HOME/sbin/stop-dfs.sh

# Start everything
$HADOOP_HOME/sbin/start-dfs.sh
$HADOOP_HOME/sbin/start-yarn.sh
$HADOOP_HOME/bin/mapred --daemon start historyserver

# Check status
jps
```

### **HDFS Commands**

```bash
# List files
hdfs dfs -ls /

# Create directory
hdfs dfs -mkdir /directory

# Copy from local to HDFS
hdfs dfs -put local-file /hdfs-path/

# Copy from HDFS to local
hdfs dfs -get /hdfs-path/file local-file

# Remove file/directory
hdfs dfs -rm /hdfs-path/file
hdfs dfs -rm -r /hdfs-path/directory

# Check disk usage
hdfs dfs -du -h /

# Check file details
hdfs dfs -stat "%n %o %r %u %g %y %b" /path/to/file
```

### **YARN Commands**

```bash
# List applications
yarn application -list

# Kill application
yarn application -kill application_id

# Check node status
yarn node -list

# View logs
yarn logs -applicationId application_id
```

---

## 🚀 **PERFORMANCE OPTIMIZATION**

### **JVM Tuning for 16GB RAM**

```bash
# Add optimal memory settings to hadoop-env.sh
cat >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh << 'EOF'

# JVM Performance Tuning for 16GB RAM
export HADOOP_HEAPSIZE_MAX=4G
export HADOOP_NAMENODE_OPTS="-Xmx2g -XX:+UseG1GC"
export HADOOP_DATANODE_OPTS="-Xmx1g -XX:+UseG1GC"
EOF

# Add YARN memory settings to yarn-env.sh
cat >> $HADOOP_HOME/etc/hadoop/yarn-env.sh << 'EOF'

# YARN Memory Configuration
export YARN_RESOURCEMANAGER_HEAPSIZE=2048
export YARN_NODEMANAGER_HEAPSIZE=1024
EOF
```

---

## 🛠️ **TROUBLESHOOTING**

### **Common Issues & Solutions**

#### **1. Services Won't Start**

```bash
# Check if ports are in use
sudo netstat -tlnp | grep -E "9000|9870|8088|8040|8042|19888"

# Kill zombie processes if needed
sudo kill -9 <PID>

# Check logs
tail -n 50 $HADOOP_HOME/logs/hadoop-hadoop-*.log
```

#### **2. HDFS Safe Mode Issues**

```bash
# Check safe mode status
hdfs dfsadmin -safemode get

# Force leave safe mode (if needed)
hdfs dfsadmin -safemode leave
```

#### **3. Disk Space Issues**

```bash
# Check disk usage
df -h
hdfs dfs -du -h /

# Clean up HDFS trash
hdfs dfs -expunge

# Clean up logs
find $HADOOP_HOME/logs -name "*.log*" -mtime +7 -delete
```

#### **4. Memory Issues**

```bash
# Check memory usage
free -h
htop

# Reduce container memory if needed (yarn-site.xml):
yarn.nodemanager.resource.memory-mb = 8192 (instead of 12288)
```

### **Emergency Recovery**

```bash
# If cluster is completely broken:

# 1. Stop all services
$HADOOP_HOME/sbin/stop-all.sh
sleep 10

# 2. Kill any remaining processes
jps | grep -v Jps | awk '{print $1}' | xargs kill -9

# 3. Clean up pid files
rm -f /tmp/hadoop-*.pid

# 4. Check and fix HDFS (CAREFUL - may lose data)
hdfs fsck /

# 5. Restart services step by step
$HADOOP_HOME/sbin/start-dfs.sh
sleep 30
$HADOOP_HOME/sbin/start-yarn.sh
sleep 30
$HADOOP_HOME/bin/mapred --daemon start historyserver
```

---

## 📊 **MAINTENANCE & MONITORING**

### **Daily Health Checks**

```bash
# Create monitoring script
cat > ~/hadoop-health-check.sh << 'EOF'
#!/bin/bash
# Hadoop Health Check Script

echo "=== Hadoop Cluster Health Check - $(date) ==="

echo "1. Service Status:"
jps

echo -e "\n2. HDFS Status:"
hdfs dfsadmin -report | head -10

echo -e "\n3. YARN Status:"
yarn node -list

echo -e "\n4. Disk Usage:"
df -h | grep -E "/$|hadoop"

echo -e "\n5. Memory Usage:"
free -h

echo -e "\n6. Recent Errors in Logs:"
find $HADOOP_HOME/logs -name "*.log" -mtime -1 -exec grep -l "ERROR" {} \;

echo "=== Health Check Complete ==="
EOF

# Make executable
chmod +x ~/hadoop-health-check.sh

# Run daily check
./hadoop-health-check.sh
```

### **Automated Backup Script**

```bash
# Create backup script
cat > ~/hdfs-backup.sh << 'EOF'
#!/bin/bash
# HDFS Backup Script

BACKUP_DIR="/home/hadoop/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

echo "Starting HDFS backup - $DATE"

# Backup critical HDFS directories
hdfs dfs -get /user $BACKUP_DIR/user_$DATE
hdfs dfs -get /tmp $BACKUP_DIR/tmp_$DATE 2>/dev/null

# Backup NameNode metadata (IMPORTANT!)
cp -r /home/hadoop/hadoop/data/namenode $BACKUP_DIR/namenode_$DATE

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -type d -name "*_*" -mtime +7 -exec rm -rf {} \;

echo "Backup completed: $BACKUP_DIR"
EOF

# Make script executable
chmod +x ~/hdfs-backup.sh
```

### **Log Rotation**

```bash
# Create log rotation config
sudo cat > /etc/logrotate.d/hadoop << 'EOF'
/home/hadoop/hadoop/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    create 644 hadoop hadoop
    postrotate
        # Restart services to release log handles if needed
    endscript
}
EOF
```

---

## 🎯 **QUICK START SUMMARY**

**For fresh installation:**

1. Update VPS & install Java ✅
2. Create hadoop user with SSH keys ✅
3. Download & configure Hadoop ✅
4. Format NameNode ✅
5. Start all services ✅
6. Test with web UIs ✅

**Daily operations:**

```bash
# Start cluster
$HADOOP_HOME/sbin/start-dfs.sh && $HADOOP_HOME/sbin/start-yarn.sh && $HADOOP_HOME/bin/mapred --daemon start historyserver

# Check status
jps && hdfs dfsadmin -report | head -10

# Stop cluster
$HADOOP_HOME/bin/mapred --daemon stop historyserver && $HADOOP_HOME/sbin/stop-yarn.sh && $HADOOP_HOME/sbin/stop-dfs.sh
```

**File operations from local Windows:**

```powershell
# Upload to HDFS (2-step)
scp -i "C:\Users\tuan\.ssh\hadoop_key" local-file.txt hadoop@167.71.203.123:/home/hadoop/
ssh -i "C:\Users\tuan\.ssh\hadoop_key" hadoop@167.71.203.123 "hdfs dfs -put local-file.txt /hdfs-path/"

# Download from HDFS (2-step)
ssh -i "C:\Users\tuan\.ssh\hadoop_key" hadoop@167.71.203.123 "hdfs dfs -get /hdfs-path/file.txt ."
scp -i "C:\Users\tuan\.ssh\hadoop_key" hadoop@167.71.203.123:/home/hadoop/file.txt ./

# Using SSH alias (after creating SSH config)
scp local-file.txt hadoop-cluster:/home/hadoop/
```

---

## 🚨 **CRITICAL REMINDERS**

1. **NEVER format NameNode after initial setup** - will lose all data
2. **Always check `jps` output** - should show 6 services running
3. **Monitor disk space** - HDFS will go into safe mode if disk > 90% full
4. **Backup NameNode metadata regularly** - critical for recovery
5. **Use `hadoop-master` hostname** in all configs, not localhost
6. **Keep firewall rules minimal** - only open necessary ports

---

**🎉 CONGRATULATIONS! Your single-node Hadoop cluster is ready!**

Access web interfaces:

- **HDFS**: http://167.71.203.123:9870
- **YARN**: http://167.71.203.123:8088
- **Jobs**: http://167.71.203.123:19888

Happy Hadoop processing! 🚀
