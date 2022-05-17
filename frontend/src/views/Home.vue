<template>
  <div>
    <h2> Ropsten TTD predictions </h2>
    <p><a href=https://github.com/taxmeifyoucan/predict_ttd/>Contribute and make your own prediction</a></p>
    <p>Total Terminal Difficulty of {{ target }} is expected around {{ msg }}</p>
    <img src="../assets/chart.png" alt="">

  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'Ping',
  data() {
    return {
      msg: '',
    };
  },
  methods: {
    getMessage() {
      axios.get('/prediction')
        .then((res) => {
          this.msg = res.data;
        })
        .catch((error) => {
          console.error(error);
        });
    },
  },
  created() {
    this.getMessage();
  },
};
</script>

