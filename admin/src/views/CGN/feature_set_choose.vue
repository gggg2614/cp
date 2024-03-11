<template>
  <div v-loading=vloading element-loading-text="处理中...">
    <el-upload
      class="upload-demo"
      :on-error="handleError"
      drag
      action="/upload"
      :on-success="uploadSuccess"
      :on-remove="uploadRemove"
      :before-upload="beforeUpload"
      :disabled="uploadDisabled"
      :limit="1"
      :file-list="files"
    >
      <i class="el-icon-upload"></i>
      <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
      (最多上传一个文件)
    </el-upload>
    <el-button type="primary" @click="submit" :disabled="submitDisabled">{{
      submitText
    }}</el-button>
    <el-button type="success" @click="downloadFile" v-if="showDownloadButton">下载结果</el-button>
  </div>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import { ElUpload, ElButton, ElMessage, ElLoading } from "element-plus";

const files = ref([]);
const submitText = ref("开始特征选择");
const uploadDisabled = ref(false);
const submitDisabled = ref(true);
const showDownloadButton = ref(false); // 控制下载按钮的显示
const vloading = ref(false);

const uploadSuccess = (res, file, filelist) => {
  files.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage({ type: "success", message: "上传成功" });
  submitDisabled.value = false; // 允许点击开始特征选择按钮
};

const beforeUpload = (file: File) => {
  if (file.name.split(".")[1] !== "xlsx") {
    ElMessage.error("请上传.xlsx文件");
    return false;
  }
};

const uploadRemove = ()=>{
  submitDisabled.value = true;
  files.value=[]
}

const submit = async () => {
  submitDisabled.value = true; // 禁用按钮
  vloading.value = true;
  if (!files.value || files.value.length === 0) {
    ElMessage.error("请上传文件。");
    return;
  }
  const formData = new FormData();
  formData.append("file", files.value[0]);
  try {
    const response = await fetch("/feature_selection", {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      ElMessage.error(`请检查文件是否有误`)
      throw new Error("网络响应不正常");
    }
    showDownloadButton.value = true
    vloading.value = false
  } catch (error) {
    submitText.value = "开始特征选择"; // 恢复按钮文本
    submitDisabled.value = false; // 允许点击按钮
    vloading.value = false
  } finally{
    submitText.value = "开始特征选择"; // 恢复按钮文本
    submitDisabled.value = false; // 允许点击按钮
    vloading.value = false
  }
};

const handleError = (error, file, fileList) => {
  ElMessage.error('上传文件发生错误，请检查文件是否正确');
};


const downloadFile = async() => {
  try {
    const response = await fetch("/download_feature_set", {
      method: "GET"
    });
    const data = await response.blob();
    const url = window.URL.createObjectURL(data);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "result.csv");
    document.body.appendChild(link);
    link.click();
    ElMessage.success("下载成功");
  } catch (error) {
    ElMessage.error("下载失败:", error);
  }
};
</script>

<style>
.upload-demo {
  margin-bottom: 20px;
}
</style>
