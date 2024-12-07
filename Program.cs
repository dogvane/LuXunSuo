using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Ollama;
using MessagePack;
using System.Numerics.Tensors;
using System.Collections;
using System.Diagnostics;
using System.ComponentModel.Design;

class Program
{
    static OllamaApiClient api = new OllamaApiClient();

    static void Main(string[] args)
    {
        var folder = "../../../data/msgpack";

        List<BookChunk> books = 加载MsgPack文件(folder);

        while (true)
        {
            Console.WriteLine("ask: 请输入鲁迅的话：");
            var say = Console.ReadLine();

            say = say.Trim();
            if (string.IsNullOrEmpty(say))
            {
                continue;
            }

            if (say == "exit")
            {
                break;
            }

            if(say == "create")
            {
                var jsonFolder = "../../../data";
                RebuildData(jsonFolder);
                continue;
            }

            BookChunk findItem = 获得文字的近似对话(books, say);

            if (findItem != null)
            {
                var ret = 检查对话是不是鲁迅说的(findItem, say);
                Console.WriteLine("对话结果："+ret);
            }
        }

    }

    private static BookChunk 获得文字的近似对话(List<BookChunk> books, string say)
    {
        var stopwatch = Stopwatch.StartNew();

        // 对输入的做编码
        float[] codes = GetEmbedding(say);
        stopwatch.Stop();

        Console.WriteLine($"编码时间:{stopwatch.Elapsed.TotalSeconds:F2}秒");

        float current_distance = float.MaxValue;
        // int current_distance = int.MaxValue;

        BookChunk findItem = null;
        stopwatch.Restart();

        foreach (var item in books)
        {
            var distance = TensorPrimitives.Distance(codes.AsSpan(), item.ChunkEmbedding.AsSpan());
            // var distance = TensorPrimitives.HammingDistance<float>(codes.AsSpan(), item.ChunkEmbedding.AsSpan());

            if (distance < current_distance)
            {
                current_distance = distance;
                findItem = item;
            }
        }

        stopwatch.Stop();
        
        Console.WriteLine($"查找时间:{stopwatch.Elapsed.TotalSeconds:F2}秒");
        Console.WriteLine($"原始文本：{findItem.Window}");

        return findItem;
    }

    /// <summary>
    /// 获得文本的编码
    /// bge-m3 的长度是1024 
    /// </summary>
    /// <param name="text"></param>
    /// <returns></returns>
    private static float[] GetEmbedding(string text)
    {
        var embed = api.Embeddings.GenerateEmbeddingAsync("bge-m3", text);
        embed.Wait();
        var codes = embed.Result.Embedding.Select(o => (float)o).ToArray();
        return codes;
    }

    static string 检查对话是不是鲁迅说的(BookChunk chunk, string 输入的对话)
    {
        var stopwatch = Stopwatch.StartNew();

        string systemPrompt = "你是鲁迅作品研究者，熟悉鲁迅的各种作品。";
        string userPrompt = (
            $"请你根据提供的参考信息，查找是否有与问题语义相似的内容。参考信息：{chunk.Window}。问题：{输入的对话}。\n" +
            $"如果找到了相似的内容，请回复“鲁迅的确说过类似的话，原文是[原文内容]，这句话来自[{chunk.Title}]”。\n" +
            $"[原文内容]是参考信息中ref字段的值，[{chunk.Title}]是参考信息中title字段的值。如果引用它们，请引用完整的内容。\n" +
            $"如果参考信息没有提供和问题相关的内容，请回答“据我所知，鲁迅并没有说过类似的话。”"
        );
        
        //Console.WriteLine(userPrompt);

        var chat = new Chat(api, "glm4:latest");
        var ret = chat.SendAsync(userPrompt);
        ret.Wait();

        stopwatch.Stop();
        Console.WriteLine($"对话检查时间:{stopwatch.Elapsed.TotalSeconds:F2}秒");

        return ret.Result.Content;        
    }

    private static List<BookChunk> 加载MsgPack文件(string folder)
    {
        Stopwatch stopwatch = new Stopwatch();
        stopwatch.Start();

        List<BookChunk> books = new List<BookChunk>();
        foreach (var file in Directory.GetFiles(folder))
        {
            if (file.EndsWith(".msgpack"))
            {
                // 从文件读取序列化数据
                byte[] readSerializedBook = File.ReadAllBytes(file);
                var datas = MessagePackSerializer.Deserialize<BookChunk[]>(readSerializedBook);

                books.AddRange(datas);
            }
        }
        stopwatch.Stop();
        Console.WriteLine($"加载msgpack时间:{stopwatch.Elapsed.TotalSeconds:F2}秒 一共 {books.Count} 条数据");

        //// 输出用于测试的代码
        //var deserializedBook = books.First();

        //// 输出反序列化后的对象
        //Console.WriteLine($"Id: {deserializedBook.Id}");
        //Console.WriteLine($"BookName: {deserializedBook.BookName}");
        //Console.WriteLine($"Title: {deserializedBook.Title}");
        //Console.WriteLine($"Author: {deserializedBook.Author}");
        //Console.WriteLine($"Type: {deserializedBook.Type}");
        //Console.WriteLine($"Source: {deserializedBook.Source}");
        //Console.WriteLine($"Date: {deserializedBook.Date}");
        //Console.WriteLine($"Chunk: {deserializedBook.Chunk}");
        //Console.WriteLine($"Window: {deserializedBook.Window}");
        //Console.WriteLine($"Method: {deserializedBook.Method}");
        //Console.WriteLine($"Method: {deserializedBook.ChunkEmbedding[0]}");
        //Console.WriteLine($"Method: {deserializedBook.WindowEmbedding[0]}");

        return books;
    }

    private const int BatchSize = 5000;
    private const int QueryBatchSize = 50;

    static void RebuildData(string jsonFolder)
    {
        var startTime = DateTime.Now;
        var inputFilePaths = Directory.GetFiles(jsonFolder, "*.json").ToList();

        foreach (var inputFilePath in inputFilePaths)
        {
            var jsonText = File.ReadAllText(inputFilePath);
            var dataList = JsonConvert.DeserializeObject<List<BookChunk>>(jsonText);

            var totalQueries = dataList.Count;
            var numBatches = (int)Math.Ceiling((double)totalQueries / BatchSize);
            Console.WriteLine($"共有 {totalQueries} 条数据，将分为 {numBatches} 批次入库。");

            Stopwatch stopwatch = new Stopwatch();

            for (int batchNum = 0; batchNum < numBatches; batchNum++)
            {
                var msgpackFilePath = inputFilePath.Replace(".json", $"_embedding_{batchNum}.msgpack");
                if (File.Exists(msgpackFilePath))
                {
                    Console.WriteLine($"第 {batchNum + 1} 批次的嵌入已经生成，跳过。");
                    continue;
                }

                var startIndex = batchNum * BatchSize;
                var endIndex = Math.Min(startIndex + BatchSize, totalQueries);
                var batchDatas = dataList.GetRange(startIndex, endIndex - startIndex);
                var batch = 0;
                stopwatch.Restart();
                foreach (var bookCheuck in batchDatas)
                {
                    if (batch++ % QueryBatchSize == 0)
                    {
                        Console.WriteLine($"第 ({batch / QueryBatchSize}/{BatchSize / QueryBatchSize}) 次 用时 {stopwatch.Elapsed.TotalSeconds:F2} 秒");
                        stopwatch.Restart();
                    }

                    bookCheuck.ChunkEmbedding = GetEmbedding(bookCheuck.Chunk);
                    bookCheuck.WindowEmbedding = GetEmbedding(bookCheuck.Window);
                }

                using (var file = new FileStream(msgpackFilePath, FileMode.Create, FileAccess.Write))
                {
                    MessagePackSerializer.Serialize(file, batchDatas);
                }

                Console.WriteLine($"第 {batchNum + 1} 批次的嵌入生成成功，共 {batchDatas.Count} 条数据。");
            }
        }

        var elapsedTime = DateTime.Now - startTime;
        Console.WriteLine($"数据入库耗时：{elapsedTime.TotalSeconds:F2} 秒");
    }
}

[MessagePackObject]
public class BookChunk
{
    [Key("id")]
    public string Id { get; set; }

    [Key("book")]
    public string BookName { get; set; }

    [Key("title")]
    public string Title { get; set; }

    [Key("author")]
    public string Author { get; set; }

    [Key("type")]
    public string Type { get; set; }

    [Key("source")]
    public string Source { get; set; }

    [Key("date")]
    public string Date { get; set; }

    [Key("chunk")]
    public string Chunk { get; set; }

    [Key("window")]
    public string Window { get; set; }

    [Key("method")]
    public string Method { get; set; }

    [Key("chunk_embedding")]
    public float[] ChunkEmbedding { get; set; }

    [Key("window_embedding")]
    public float[] WindowEmbedding { get; set; }

}
