const MaxWorkGroupSize = 256;
const adapter = await navigator.gpu?.requestAdapter();
const device = await adapter?.requestDevice();

if (!device) {
  fail("need a browser that supports WebGPU");
}
const module = device.createShaderModule({
  label: "reducer",
  code: `
  @group(0) @binding(0) var<storage,read> inputBuffer: array<u32>;
  @group(0) @binding(1) var<storage,read_write> output: atomic<u32>;
  
  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) id: vec3<u32>) {
      // Accumulate in buffer:
      atomicAdd(&output, inputBuffer[id.x]);
  }`,
});
const pipeline = device.createComputePipeline({
  label: "sum",
  layout: "auto",
  compute: {
    module,
  },
});
const outputSize = 4;
const outputStorageBuffer = device.createBuffer({
  label: "outputStorageBuffer",
  size: outputSize,
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});

const radomValues = [];
for (let i = 0; i < 4000000; i++) {
  radomValues[i] = Math.floor(Math.random() * 1000);
}

export const sumValues = new Int32Array(radomValues);

const sumValuesBuffer = device.createBuffer({
  label: "sum values",
  size: sumValues.length * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(sumValuesBuffer, 0, sumValues);

const resBuffer = device.createBuffer({
  label: "resBuffer",
  size: outputStorageBuffer.size,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});

const bindGroup = device.createBindGroup({
  label: "sum bind group",
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    {
      binding: 0,
      resource: {
        buffer: sumValuesBuffer,
      },
    },
    {
      binding: 1,
      resource: {
        buffer: outputStorageBuffer,
      },
    },
  ],
});

const encoder = device.createCommandEncoder();
const computePass = encoder.beginComputePass();

computePass.setBindGroup(0, bindGroup);
computePass.setPipeline(pipeline);

computePass.dispatchWorkgroups(
  Math.ceil(sumValuesBuffer.size / MaxWorkGroupSize),
  1
);
computePass.end();

encoder.copyBufferToBuffer(
  outputStorageBuffer,
  0,
  resBuffer,
  0,
  outputStorageBuffer.size
);

console.time("gpu0 execution");
device.queue.submit([encoder.finish()]);

let sum = 0;
console.time("cpu execution");
radomValues.forEach((r) => {
  sum += r;
});

console.log(sum);
console.timeEnd("cpu execution");

// Finally, map and read from the CPU-readable buffer.
resBuffer
  .mapAsync(
    GPUMapMode.READ,
    0, // Offset
    outputStorageBuffer.length // Length
  )
  .then(() => {
    //resolves the Promise created by the call to mapAsync()
    const copyArrayBuffer = resBuffer.getMappedRange(
      0,
      outputStorageBuffer.length
    );
    const data = copyArrayBuffer.slice();
    resBuffer.unmap();

    console.log(new Int32Array(data)[0]); //display the information.
    console.timeEnd("gpu0 execution");
  });
