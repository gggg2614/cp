<template>
  <div>
    <el-steps :active="activeStep" finish-status="success" align-center style="margin-bottom: 20px;">
      <el-step title="第一步"></el-step>
      <el-step title="第二步"></el-step>
      <el-step title="第三步"></el-step>
    </el-steps>

    <el-card v-if="activeStep === 0">
      <el-form label-width="120px" :model="form">
        <el-form-item label="输入搜索次数" prop="nSearches">
          <el-input v-model="nSearches" placeholder="输入搜索次数" type="number" min="0"></el-input>
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="nextStep">下一步</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-upload v-if="activeStep === 1" drag :action="importURL" :on-success="uploadSuccess" :before-upload="beforeUpload"
      :auto-upload="true">
      <el-button slot="trigger" size="small" type="primary">选择文件</el-button>
      <div class="el-upload__tip" slot="tip">支持上传 XLSX/XLS 文件</div>
    </el-upload>

    <el-card v-if="activeStep === 2">
      <h3>返回的结果：</h3>
      <el-form label-width="200" v-if="Object.keys(form).length > 0">
        <el-row justify="center">
          <el-col v-for="(value, key) in form" :key="key">
            <el-form-item :label="key">
              <el-input v-model="form[key]" :readonly="true"></el-input>
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
      <el-button @click="prevStep" type="primary" size="small">返回上一步</el-button>
      <el-button @click="downloadCSV" type="primary" size="small" :loading="csvLoading">下载结果 CSV</el-button>

    </el-card>


    <el-button v-if="activeStep === 1" @click="submit" type="primary" size="small" :loading="subLoading">提交</el-button>
    <el-button v-show="activeStep === 1" @click="prevStep" slot="tip" size="small">返回</el-button>
  </div>
</template>

<script lang="ts" setup>
import { ref, nextTick, watchEffect } from 'vue';
import { ElMessage } from 'element-plus';

const activeStep = ref(0);
const nSearches = ref('');
const files = ref([]);
const form = ref({});
const subLoading = ref(false);
const csvLoading = ref(false);
const importURL = 'http://localhost:5000/upload';

const handleBackendResponse = (data) => {
  const bestParams = data.best_params;
  console.log("bestParams: ", bestParams);
  form.value = {}; // 清空 form
  for (const key in bestParams) {
    form.value[key] = bestParams[key];
  }
  form.value['best_score'] = data.best_score;
  console.log("form after assignment: ", form);
};

const nextStep = () => {
  if (parseInt(nSearches.value) <= 0) {
    ElMessage.error('请输入大于0的整数')
    return;
  }
  (async () => {
    await nextTick();
    if (nSearches.value.trim() === '') {
      ElMessage.error('请输入搜索次数。');
      return;
    }
    if (!/^\d+$/.test(nSearches.value)) {
      ElMessage.error('搜索次数必须为整数。');
      return;
    }
    activeStep.value++;
  })();
};

const prevStep = () => {
  activeStep.value--;
  files.value = []
};

const beforeUpload = (file: File) => {
  if (file.name.split(".")[1] !== "xlsx") {
    ElMessage.error("请上传xlsx");
    return false;
  }
};

const uploadSuccess = (res, file, filelist) => {
  files.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage({ type: "success", message: "上传成功" });
};

const downloadCSV = () => {
  fetch('http://localhost:5000/generate_features', {
    method: 'POST',
    body: JSON.stringify({ best_params: form.value }), // 将前端获取的 form 传给后端
    headers: {
      'Content-Type': 'application/json'
    }
  }).then(response => {
    if (!response.ok) {
      throw new Error('网络响应不正常');
    }
    return response.blob();
  }).then(blob => {
    const url = window.URL.createObjectURL(new Blob([blob]));
    const link = document.createElement('a');
    csvLoading.value = true;
    link.href = url;
    link.setAttribute('download', 'features.csv'); // 设置下载的文件名为 features.csv
    document.body.appendChild(link);
    link.click();
    link.parentNode.removeChild(link);
  }).catch(error => {
    console.error('发生了与 fetch 操作相关的问题：', error);
  }).finally(()=>{
    csvLoading.value = false;
  });
};

const submit = () => {
  if (!files.value || files.value.length === 0) {
    ElMessage.error('请上传文件。');
    return;
  }
  const formData = new FormData();
  formData.append('file', files.value[0]);
  formData.append('n_searches', nSearches.value);
  
  subLoading.value = true;
  fetch('http://localhost:5000/file1', {
    method: 'POST',
    body: formData
  }).then(response => {
    if (!response.ok) {
      throw new Error('网络响应不正常');
    }
    return response.json();
  }).then(data => {
    handleBackendResponse(data);
    activeStep.value = 2;
  }).catch(error => {
    console.error('发生了与 fetch 操作相关的问题：', error);
  }).finally(() => {
    subLoading.value = false;
  });

};
</script>
