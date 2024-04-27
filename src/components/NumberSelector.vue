<script setup lang="ts">
const { count } = defineProps<{
  count: number
  color: string
}>()
const num = defineModel<number>({ required: true })

const setNum = (n: number) => {
  if (num.value === n) {
    num.value--
  } else {
    num.value = n
  }
}
</script>

<template>
  <div>
    <template v-for="index in count" :key="index">
      <label>
        <input type="radio" @click="setNum(index)" :checked="num >= index" />
        <span class="checkmark"></span>
      </label>
    </template>
  </div>
</template>

<style scoped>
label {
  margin-right: 2vw;
  user-select: none;
}

input[type='radio'] {
  display: none;
}

/* 自定义radio的样式 */
.checkmark {
  position: relative;
  display: inline-block;
  width: 16vh; /* 根据需要调整大小 */
  height: 16vh; /* 根据需要调整大小 */
  border: 0.06em solid currentColor; /* 控制轮廓颜色和宽度 */
  border-radius: 50%;
  background-color: transparent; /* 默认透明 */
  vertical-align: middle; /* 使文字与radio对齐 */
  cursor: pointer; /* 改变鼠标指针 */
}

/* 当radio被选中时改变样式 */
input[type='radio']:checked ~ .checkmark {
  background-color: v-bind(color);
}

/* 内部小圆点 */
input[type='radio']:checked ~ .checkmark::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 5vh;
  height: 5vh;
  border-radius: 50%;
  background-color: darkorange;
}
</style>
