export default {
  path: "/student",
  redirect: "/student/List",
  meta: {
    icon: "twemoji:man-scientist-dark-skin-tone",
    title: "实验",
    // showLink: false,
    rank: 1,
  },
  children: [
    {
      path: "/student/add",
      name: "studentAdd",
      showParent: true,
      component: () => import("@/views/student/studentAdd.vue"),
      meta: {
        title: "新增"
      }
    },
    {
      path: "/student/list",
      name: "studentList",
      showParent: true,
      component: () => import("@/views/student/studentList.vue"),
      meta: {
        title: "毕业"
      }
    }
  ]
} as unknown as RouteConfigsTable;
