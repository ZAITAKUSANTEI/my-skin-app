// Google Cloudのライブラリと、画像アップロードを処理するためのライブラリをインポート
const { VertexAI } = require('@google-cloud/vertexai');
const { ImageAnnotatorClient } = require('@google-cloud/vision');
const multipart = require('lambda-multipart-parser');

// Netlifyのメイン関数
exports.handler = async function(event) {
    // --- 認証情報とAIクライアントの初期化 ---
    // この関数が呼び出されるたびに、認証情報をチェックし、AIクライアントを準備します。
    try {
        const gcpSaKeyBase64 = process.env.GCP_SA_KEY_BASE64;

        if (!gcpSaKeyBase64) {
            console.error("Fatal Error: GCP_SA_KEY_BASE64 environment variable not found.");
            throw new Error("サーバーの設定エラーです。環境変数が設定されていません。");
        }

        const credentialsJson = Buffer.from(gcpSaKeyBase64, 'base64').toString('utf8');
        const credentials = JSON.parse(credentialsJson);
        const projectId = credentials.project_id;

        if (!credentials.client_email || !credentials.private_key) {
            console.error("Authentication Error: Service account credentials not parsed correctly.");
            throw new Error("サービスアカウントの認証情報が正しく設定されていません。");
        }
        console.log(`Successfully parsed credentials for service account: ${credentials.client_email}`);

        // ★★★ 修正点：認証方法をより具体的に指定 ★★★
        const explicitCredentials = {
            client_email: credentials.client_email,
            private_key: credentials.private_key,
        };

        // Vertex AI (Gemini)
        const vertexAI = new VertexAI({ project: projectId, location: 'asia-northeast1', credentials: explicitCredentials });
        const generativeModel = vertexAI.getGenerativeModel({ model: 'gemini-1.5-flash-001' });

        // Vision API
        const visionClient = new ImageAnnotatorClient({ credentials: explicitCredentials });
        
        // --- ここからがメインの処理 ---
        if (event.httpMethod !== 'POST') {
            return { statusCode: 405, body: 'Method Not Allowed' };
        }

        // 1. フロントエンドから画像データを受け取る
        const result = await multipart.parse(event);
        const frontImage = result.files.find(f => f.fieldname === 'frontImage');
        if (!frontImage) {
            throw new Error('正面画像が見つかりません。');
        }

        // 2. Vision APIで顔を分析する
        const [visionResult] = await visionClient.faceDetection(frontImage.content);
        const faceAnnotations = visionResult.faceAnnotations;
        if (!faceAnnotations || faceAnnotations.length === 0) {
            throw new Error('顔を検出できませんでした。');
        }
        const face = faceAnnotations[0];

        // 3. 評価基準に基づいてスコアを算出する
        const scores = calculateScores(face);

        // 4. Vertex AI (Gemini) への指示書を作成する
        const prompt = createPrompt(scores, face);

        // 5. Vertex AI (Gemini) で提案レポートを生成する
        const geminiResult = await generativeModel.generateContent(prompt);
        const reportHtml = geminiResult.response.candidates[0].content.parts[0].text;

        // 6. フロントエンドに結果を返す
        return {
            statusCode: 200,
            body: JSON.stringify({ reportHtml, scores }),
        };

    } catch (error) {
        console.error("AI analysis failed:", error);
        return {
            statusCode: 500,
            body: JSON.stringify({ message: error.message || "サーバーで不明なエラーが発生しました。" }),
        };
    }
};

// --- ヘルパー関数 ---

function calculateScores(face) {
    const likelihoods = ['UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY'];
    const joyScore = likelihoods.indexOf(face.joyLikelihood) * 20;
    const sorrowScore = likelihoods.indexOf(face.sorrowLikelihood) * 20;
    const smoothness = 100 - ((joyScore + sorrowScore) / 2);
    const tiltAngleScore = 100 - Math.abs(face.tiltAngle);
    const surpriseScore = likelihoods.indexOf(face.surpriseLikelihood) * 20;
    const firmness = (tiltAngleScore + (100 - surpriseScore)) / 2;
    const underExposedScore = likelihoods.indexOf(face.underExposedLikelihood) * 20;
    const dullness = 100 - underExposedScore;
    const blurredScore = likelihoods.indexOf(face.blurredLikelihood) * 20;
    const spots = 100 - blurredScore;
    const pores = 100 - blurredScore;
    const finalize = (score) => Math.max(0, Math.min(Math.round(score), 100));
    return {
        dullness: finalize(dullness),
        smoothness: finalize(smoothness),
        firmness: finalize(firmness),
        spots: finalize(spots),
        pores: finalize(pores),
    };
}

function createPrompt(scores, face) {
    const treatmentsDB = `カテゴリ,治療法名,特徴,価格（円）
しわ,ボトックス,表情ジワの改善に即効性あり,20000
しわ,ヒアルロン酸注入,ボリュームアップに効果的,50000
しわ,PRP注入,自己血液を使った自然な再生治療,60000
しわ,スレッドリフト,引き上げ効果が高い,80000
しわ,マイクロニードルRF,皮膚深層への刺激でコラーゲン生成,70000
たるみ,HIFU,超音波で筋膜にアプローチ,90000
たるみ,スレッドリフト,糸による物理的なリフト,85000
たるみ,RF（高周波）,熱による皮膚の引き締め,60000
たるみ,ウルセラ,FDA認可のたるみ治療,120000
たるみ,サーマクール,高周波による深部加熱,100000
毛穴,フラクショナルCO2レーザー,レーザーで毛穴と皮膚再生を促進,40000
毛穴,ダーマペン,微細針でコラーゲン生成促進,30000
毛穴,ポテンツァ,微細針＋高周波で毛穴改善,80000
毛穴,ハイドラフェイシャル,毛穴と角質のディープクレンジング,20000
毛穴,カーボンピーリング,炭を用いたピーリングで引き締め,25000
赤み,IPL（フォトフェイシャル）,光による赤み・色ムラの改善,35000
赤み,Vビームレーザー,血管に特化した赤ら顔治療,45000
赤み,ロゼックスゲル,酒さ・赤ら顔に使用,3000
赤み,赤外線治療,赤外線で血行促進,15000
赤み,フラクショナルレーザー,赤みと同時に肌質も改善,60000
色素沈着,トラネキサム酸内服,肝斑・色素沈着に内服,5000
色素沈着,レーザートーニング,レーザーで均一な肌トーンに,30000
色素沈着,ハイドロキノン,メラニン抑制クリーム,4000
色素沈着,ルメッカ,しみやそばかすの改善,35000
色素沈着,ピコトーニング,肝斑に適した微弱レーザー,45000
肌質改善,エレクトロポレーション,成分導入による肌質改善,15000
肌質改善,マッサージピール,皮むけを伴うリフト＆美白,20000
肌質改善,プラズマ治療,殺菌・肌再生効果あり,40000
肌質改善,エクソソーム導入,幹細胞成分による肌修復,70000
肌質改善,水光注射,保湿と美容成分注入,30000
脂肪除去,脂肪溶解注射,脂肪細胞を直接分解,20000
脂肪除去,クールスカルプティング,冷却で脂肪細胞破壊,90000
脂肪除去,脂肪吸引,外科的な脂肪除去,300000
脂肪除去,HIFU（脂肪層）,脂肪層にピンポイント照射,100000
脂肪除去,カベリン注射,植物由来成分の脂肪融解,25000`;
    const visionAnalysisSummary = `
- 喜びの可能性: ${face.joyLikelihood}
- 悲しみの可能性: ${face.sorrowLikelihood}
- 驚きの可能性: ${face.surpriseLikelihood}
- 露出不足の可能性: ${face.underExposedLikelihood}
- ぼやけの可能性: ${face.blurredLikelihood}
- 顔の傾き: ${face.tiltAngle.toFixed(2)}度
`;
    return `
あなたは日本で最も信頼されている美容カウンセラーです。
以下の2つの情報をもとに、クライアントにパーソナライズされた美容プランを提案してください。

# 情報1：AIによる肌スコア (100点満点)
- くすみ: ${scores.dullness}
- なめらかさ(しわ): ${scores.smoothness}
- ハリ(たるみ): ${scores.firmness}
- シミ: ${scores.spots}
- 毛穴: ${scores.pores}

# 情報2：Google Vision APIによる詳細な顔分析データ
${visionAnalysisSummary}

# 提案可能な治療法リスト
${treatmentsDB}

# あなたへの指示
1. まず「AIによる診断結果」として、スコアと分析データを基に、クライアントの肌の状態を総合的に評価してください。特にスコアが低い項目について言及してください。
2. 次に「あなたへの最適な治療プラン」として、スコアが低い悩みを解決するために、治療法リストの中から最も関連性の高い治療法を2つずつ提案してください。
3. 提案する際は、「治療法名」「特徴」「参考価格」を分かりやすくまとめてください。価格には「円」を付けてください。
4. 全体を通して、専門的でありながらも、利用者に寄り添うような温かいトーンで記述してください。
5. 回答は必ずHTML形式で出力してください。診断結果のタイトルは<h3>タグ、各治療法の提案は<div>で囲み、治療法名は<h5>タグ、特徴と価格は<p>タグを使用してください。
`;
}