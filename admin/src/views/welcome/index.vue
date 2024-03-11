<template>
  <div class="container">
    <h1 class="title">欢迎使用数据分析系统</h1>
  </div>
</template>

<style scoped>
.container {
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 50vh;
}

.title {
  font-size: 3rem;
  color: #333;
  animation: bounce 1s infinite alternate, color-change 5s infinite;
  transition: text-shadow 0.3s ease-in-out;
}

.title:hover {
  text-shadow: 0 0 10px rgba(0, 0, 0, 0.5); /* 调整阴影效果 */
}

@keyframes bounce {
  0% {
    transform: translateY(0);
  }
  100% {
    transform: translateY(-20px);
  }
}

@keyframes color-change {
  0% {
    color: #333;
  }
  25% {
    color: #ff0000;
  }
  50% {
    color: #00ff00;
  }
  75% {
    color: #0000ff;
  }
  100% {
    color: #333;
  }
}

@keyframes smoke {
  0% {
    opacity: 0.6;
    transform: scale(0.9);
  }
  50% {
    opacity: 0.8;
    transform: scale(1.1);
  }
  100% {
    opacity: 0.6;
    transform: scale(0.9);
  }
}
</style>
