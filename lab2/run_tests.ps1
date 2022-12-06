$images=@('1024x768.png', '1280x960.png', '2048x1536.png', '8192x8192.png')

foreach ($image in $images) {
    for ($i = 0; $i -lt 10; $i++) {
        python main.py --input "../images/$image" --output "./output/test.png"
    }
}
