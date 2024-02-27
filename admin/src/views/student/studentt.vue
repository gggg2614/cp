<template>
  <el-container>
    <el-main>
      <el-form ref="form" :model="formData" label-width="120px">
        <el-form-item label="Cycles">
          <el-input v-model.trim="formData.cycles" placeholder="请输入整数" min="0" type="number" ></el-input>
        </el-form-item>
        <el-form-item label="Epoch">
          <el-input v-model.trim="formData.epoch" placeholder="请输入整数" min="0" type="number" ></el-input>
        </el-form-item>
        <el-form-item label="FeatureAddNum">
          <el-input v-model.trim="formData.feature_add_num" placeholder="请输入0-10的整数" type="number" min="0" max="10"></el-input>
        </el-form-item>
        <el-form-item label="Data File(.xlsx)">
          <el-upload action="http://localhost:5000/upload" :on-success="handleDataUpload">
            <el-button slot="trigger" type="primary">Upload</el-button>
          </el-upload>
        </el-form-item>
        <el-form-item label="Data2 File(.csv)">
          <el-upload :on-success="handleData2Upload" action="http://localhost:5000/upload">
            <el-button slot="trigger" type="primary">Upload</el-button>
          </el-upload>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="validateAndTrain" :loading="trainLoading" :disabled="!isFormValid">Train
            Model</el-button>
        </el-form-item>
        <el-form-item v-if="trainComplete">
          <el-button type="success" @click="downloadCsv">Download CSV</el-button>
          <el-button type="success" @click="downloadPth">Download PTH</el-button>
        </el-form-item>
      </el-form>
    </el-main>
  </el-container>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { ElMessage } from 'element-plus';

const formData = ref({
  cycles: '',
  epoch: '',
  feature_add_num: '',
});

const data = ref([])
const data2 = ref([])
const trainLoading = ref(false);
const trainComplete = ref(false);

const validateAndTrain = async () => {
  const featureAddNum = parseInt(formData.value.feature_add_num);
  if (isNaN(featureAddNum) || featureAddNum < 0 || featureAddNum > 10) {
    console.log(featureAddNum);
    ElMessage.error('Feature Add Num must be a number between 0 and 10');
    return;
  }
  if (data.value.length === 0 || data2.value.length === 0) { 
    ElMessage.error('请上传文件')
    return;
  }
  trainModel();
};

const handleDataUpload = (response: any, file, filelist) => {
  data.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage.success('Data file uploaded successfully');
}

const handleData2Upload = (response: any, file, filelist) => {
  data2.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage.success('Data2 file uploaded successfully');
};

const trainModel = async () => {
  if (!isFormValid) return;
  trainLoading.value = true;
  const formData1 = new FormData();
  formData1.append('cycles', formData.value.cycles);
  formData1.append('epoch', formData.value.epoch);
  formData1.append('feature_add_num', formData.value.feature_add_num);
  formData1.append('data', data.value[0]);
  formData1.append('data2', data2.value[0]);

  try {
    const response = await fetch('http://localhost:5000/train', {
      method: 'POST',
      body: formData1,
    });
    const data = await response.blob();
    if (data) {
      ElMessage.success('Model trained successfully');
      trainComplete.value = true;
    } else {
      ElMessage.error('Failed to train model');
    }
  } catch (error) {
    console.error('Error:', error);
    ElMessage.error('Failed to train model');
  } finally {
    trainLoading.value = false;
  }
};

const downloadCsv = async () => {
  try {
    const response = await fetch('http://localhost:5000/download_results', {
      method: 'GET',
    });
    const data = await response.blob();
    const url = window.URL.createObjectURL(data);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'results.csv');
    document.body.appendChild(link);
    link.click();
    ElMessage.success('Results CSV downloaded successfully');
  } catch (error) {
    console.error('Error:', error);
    ElMessage.error('Failed to download results CSV');
  }
};

const downloadPth = async () => {
  try {
    const response = await fetch('http://localhost:5000/download_model', {
      method: 'GET',
    });
    const data = await response.blob();
    const url = window.URL.createObjectURL(data);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'best_model.pth');
    document.body.appendChild(link);
    link.click();
    ElMessage.success('Model PTH downloaded successfully');
  } catch (error) {
    console.error('Error:', error);
    ElMessage.error('Failed to download model PTH');
  }
};
const isFormValid = computed(() => {
  return formData.value.cycles && formData.value.epoch && formData.value.feature_add_num;
});
</script>
