import { sumValues } from "./main.js";
const MaxWorkGroupSize = 256;
const adapter = await navigator.gpu?.requestAdapter();
const device = await adapter?.requestDevice();

if (!device) {
  fail("need a browser that supports WebGPU");
}
const module = device.createShaderModule({
  label: "reducer1",
  code: `
  // Reduction #4 from nvidia preduction paper:

  @group(0) @binding(0) var<storage,read> inputBuffer: array<u32>;
  @group(0) @binding(1) var<storage,read_write> output: atomic<u32>;
  
  // Create zero-initialized workgroup shared data
  const wgsize : u32 = 256;
  var<workgroup> sdata: array<u32, wgsize>;
  
  @compute @workgroup_size(wgsize)
  fn main(@builtin(workgroup_id) gid: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  
      // Each thread should read its data:
      var tid: u32 = local_id.x;
      // Scale the groupsize by 2 below to get the overall index:
      var idx: u32 = gid.x * wgsize * 2 + tid;
  
      // Add 2 elements already:
      // Note: we don't need to check the bounds if using an input array size that can be divided by wgsize
      // sdata[tid] = select(0, inputBuffer[idx], idx < 4194304) + select(0, inputBuffer[idx + wgsize], (idx + wgsize) < 4194304);
      sdata[tid] = inputBuffer[idx] + inputBuffer[idx + wgsize];
  
      // sync all the threads:
      workgroupBarrier();
  
      // Do the reduction in shared memory:
      for (var s: u32 = wgsize / 2; s > 0; s >>= 1) {
          if tid < s {
              sdata[tid] += sdata[tid + s];
          }
          workgroupBarrier();
      }
  
  
      // Add result from the workgroup to the output storage:
      if tid == 0 {
          atomicAdd(&output, sdata[0]);
      }
  }
  
  `,
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

// const sumValues = new Int32Array(radomValues);

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

console.time("gpu4 execution");
device.queue.submit([encoder.finish()]);

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
    console.timeEnd("gpu4 execution");
  });
