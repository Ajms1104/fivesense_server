package com.example.fivesense.service;

import com.theokanning.openai.completion.chat.ChatCompletionRequest;
import com.theokanning.openai.completion.chat.ChatFunction;
import com.theokanning.openai.completion.chat.ChatFunctionCall;
import com.theokanning.openai.service.OpenAiService;
import com.example.fivesense.model.ChatList;
import com.example.fivesense.repository.ChatMessageRepository;
import com.example.fivesense.service.KiwoomApiService;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

@Service
public class ChatGPTService {

    private final OpenAiService openAiService;
    private final ChatMessageRepository chatMessageRepository;
    private final KiwoomApiService kiwoomApiService;

    public ChatGPTService(@Value("${openai.api.key}") String apiKey, 
                         ChatMessageRepository chatMessageRepository,
                         KiwoomApiService kiwoomApiService) {
        System.out.println("=== ChatGPTService 초기화 시작 ===");
        System.out.println("주입된 API 키: " + (apiKey != null ? apiKey.substring(0, Math.min(apiKey.length(), 10)) + "..." : "null"));
        System.out.println("API 키 길이: " + (apiKey != null ? apiKey.length() : 0));
        
        if (apiKey == null || apiKey.equals("your-api-key-here")) {
            System.err.println("=== API 키 설정 오류 ===");
            System.err.println("API 키가 null이거나 기본값으로 설정되어 있습니다.");
            throw new IllegalArgumentException("OpenAI API 키가 설정되지 않았습니다.");
        }
        
        System.out.println("API 키 유효성 검사 통과");
        this.openAiService = new OpenAiService(apiKey, Duration.ofSeconds(60));
        System.out.println("OpenAiService 생성 완료");
        this.chatMessageRepository = chatMessageRepository;
        this.kiwoomApiService = kiwoomApiService;
        System.out.println("ChatMessageRepository 주입 완료");
        System.out.println("KiwoomApiService 주입 완료");
        System.out.println("=== ChatGPTService 초기화 완료 ===");
    }

    public String getChatResponse(String userMessage, Long roomId) {
        System.out.println("=== ChatGPT API 호출 시작 ===");
        System.out.println("사용자 메시지: " + userMessage);
        System.out.println("채팅방 ID: " + roomId);
        
        try {
            List<com.theokanning.openai.completion.chat.ChatMessage> messages = new ArrayList<>();
            
            // 시스템 메시지 추가
            messages.add(new com.theokanning.openai.completion.chat.ChatMessage(
                "system", 
                "당신은 주식 투자 전문가입니다. 사용자의 질문에 대해 전문적이고 정확한 답변을 제공해주세요. " +
                "주식 데이터나 뉴스 데이터가 필요한 경우, 제공된 함수를 사용하여 최신 정보를 조회한 후 답변해주세요."
            ));
            System.out.println("시스템 메시지 추가 완료");
            
            // 이전 대화 내역 추가 (최근 3개 메시지만 - 토큰 제한 고려)
            List<ChatList> previousMessages = chatMessageRepository.findRecentMessagesByRoomId(roomId, 3);
            System.out.println("이전 대화 내역 개수: " + previousMessages.size());
            for (ChatList msg : previousMessages) {
                String role = "USER".equals(msg.getType()) ? "user" : "assistant";
                // 너무 긴 메시지는 잘라내기 (최대 1000자)
                String content = msg.getContent();
                if (content.length() > 1000) {
                    content = content.substring(0, 1000) + "... (생략)";
                }
                messages.add(new com.theokanning.openai.completion.chat.ChatMessage(role, content));
            }
            
            // 현재 사용자 메시지 추가
            messages.add(new com.theokanning.openai.completion.chat.ChatMessage("user", userMessage));
            System.out.println("총 메시지 개수: " + messages.size());

            // Function calling을 위한 함수 정의
            List<ChatFunction> functions = createFunctions();
            
            ChatCompletionRequest request = ChatCompletionRequest.builder()
                    .model("gpt-3.5-turbo")
                    .messages(messages)
                    .functions(functions)
                    .functionCall(ChatCompletionRequest.ChatCompletionRequestFunctionCall.of("auto"))
                    .maxTokens(1500)  // 응답 길이 제한 (너무 크면 비용 증가 및 속도 저하)
                    .temperature(0.7)
                    .build();
            System.out.println("ChatCompletionRequest 생성 완료");

            System.out.println("OpenAI API 호출 중...");
            var completion = openAiService.createChatCompletion(request);
            var message = completion.getChoices().get(0).getMessage();
            
            // Function calling 처리
            if (message.getFunctionCall() != null) {
                System.out.println("Function calling 감지: " + message.getFunctionCall().getName());
                return handleFunctionCall(message.getFunctionCall(), messages, userMessage, roomId);
            }
            
            String response = message.getContent();
            System.out.println("OpenAI API 응답 수신 완료");
            System.out.println("=== ChatGPT API 호출 성공 ===");
            
            return response;
        } catch (Exception e) {
            System.err.println("=== ChatGPT API 호출 오류 ===");
            System.err.println("오류 메시지: " + e.getMessage());
            System.err.println("오류 타입: " + e.getClass().getSimpleName());
            e.printStackTrace();
            throw new RuntimeException("ChatGPT API 호출 중 오류가 발생했습니다: " + e.getMessage());
        }
    }
    
    // 채팅방별 메시지 조회
    public List<ChatList> getChatHistory(Long roomId) {
        return chatMessageRepository.findByRoomIdOrderByTimestampAsc(roomId);
    }
    
    // 새로운 채팅방 생성 (첫 번째 메시지)
    public ChatList createNewChat(String userMessage) {
        // 새로운 roomId 생성 (현재 시간을 기반으로)
        Long newRoomId = System.currentTimeMillis();
        
        // 사용자 메시지 저장
        ChatList userMsg = new ChatList();
        userMsg.setContent(userMessage);
        userMsg.setType("USER");
        userMsg.setRoomId(newRoomId);
        chatMessageRepository.save(userMsg);
        
        // AI 응답 생성
        String aiResponse = getChatResponse(userMessage, newRoomId);
        
        // AI 메시지 저장
        ChatList aiMsg = new ChatList();
        aiMsg.setContent(aiResponse);
        aiMsg.setType("AI");
        aiMsg.setRoomId(newRoomId);
        chatMessageRepository.save(aiMsg);
        
        return aiMsg;
    }
    
    // 기존 채팅방에 메시지 추가
    public ChatList addMessageToChat(String userMessage, Long roomId) {
        // 사용자 메시지 저장
        ChatList userMsg = new ChatList();
        userMsg.setContent(userMessage);
        userMsg.setType("USER");
        userMsg.setRoomId(roomId);
        chatMessageRepository.save(userMsg);
        
        // AI 응답 생성
        String aiResponse = getChatResponse(userMessage, roomId);
        
        // AI 메시지 저장
        ChatList aiMsg = new ChatList();
        aiMsg.setContent(aiResponse);
        aiMsg.setType("AI");
        aiMsg.setRoomId(roomId);
        chatMessageRepository.save(aiMsg);
        
        return aiMsg;
    }
    
    /**
     * Function calling을 위한 함수 정의 생성
     */
    private List<ChatFunction> createFunctions() {
        List<ChatFunction> functions = new ArrayList<>();
        
        // 주식 차트 조회 함수
        ChatFunction getStockChartFunction = ChatFunction.builder()
                .name("get_stock_chart")
                .description(
                    "주식 차트 데이터를 조회합니다. " +
                    "파라미터: " +
                    "1) stock_code (필수, string): 주식 종목 코드 6자리. 반드시 숫자 6자리여야 합니다. " +
                    "   예시: 삼성전자=005930, SK하이닉스=000660, 카카오=035720, NAVER=035420, 현대차=005380 " +
                    "2) base_date (선택, string): 조회 기준일 yyyyMMdd 형식. 예: 20240101. 생략 시 오늘 날짜 " +
                    "3) api_id (선택, string): API ID. 기본값 KA10081 사용"
                )
                .executor(StockChartRequest.class, request -> {
                    // 이 부분은 실제로 실행되지 않고, 타입 정의용
                    return null;
                })
                .build();
        
        // 거래량 상위 종목 조회 함수
        ChatFunction getTopVolumeStocksFunction = ChatFunction.builder()
                .name("get_top_volume_stocks")
                .description(
                    "거래량 상위 종목을 조회합니다. " +
                    "당일 거래량이 많은 주식들의 목록을 확인할 수 있습니다. " +
                    "파라미터가 필요 없습니다."
                )
                .build();
        
        // 최신 뉴스 조회 함수
        ChatFunction getLatestNewsFunction = ChatFunction.builder()
                .name("get_latest_news")
                .description(
                    "최신 뉴스를 조회합니다. " +
                    "파라미터: " +
                    "1) page (선택, integer): 조회할 페이지 번호. 기본값 1"
                )
                .executor(NewsPageRequest.class, request -> {
                    return null;
                })
                .build();
        
        // 키워드 뉴스 검색 함수
        ChatFunction searchNewsFunction = ChatFunction.builder()
                .name("search_news")
                .description(
                    "특정 키워드로 뉴스를 검색합니다. " +
                    "파라미터: " +
                    "1) keyword (필수, string): 검색할 키워드. 회사명, 종목명, 산업 분야 등 " +
                    "   예: '삼성전자', '반도체', '전기차' 등 " +
                    "2) page (선택, integer): 조회할 페이지 번호. 기본값 1"
                )
                .executor(NewsSearchRequest.class, request -> {
                    return null;
                })
                .build();
        
        functions.add(getStockChartFunction);
        functions.add(getTopVolumeStocksFunction);
        functions.add(getLatestNewsFunction);
        functions.add(searchNewsFunction);
        
        return functions;
    }
    
    // Function calling을 위한 Request 클래스들
    public static class StockChartRequest {
        public String stock_code;
        public String base_date;
        public String api_id;
    }
    
    public static class NewsPageRequest {
        public Integer page;
    }
    
    public static class NewsSearchRequest {
        public String keyword;
        public Integer page;
    }
    
    /**
     * Function call 처리
     */
    private String handleFunctionCall(ChatFunctionCall functionCall, List<com.theokanning.openai.completion.chat.ChatMessage> messages, String userMessage, Long roomId) {
        try {
            System.out.println("=== Function Call 처리 시작 ===");
            System.out.println("함수명: " + functionCall.getName());
            System.out.println("인수: " + functionCall.getArguments());
            
            // 함수 실행 결과를 메시지에 추가
            String functionResult = executeFunction(functionCall);
            
            // 함수 실행 결과를 assistant 메시지로 추가
            messages.add(new com.theokanning.openai.completion.chat.ChatMessage("assistant", functionResult));
            
            // 최종 응답을 위한 추가 API 호출
            ChatCompletionRequest followUpRequest = ChatCompletionRequest.builder()
                    .model("gpt-3.5-turbo")
                    .messages(messages)
                    .maxTokens(1200)  // Function call 결과를 포함한 응답 생성
                    .temperature(0.7)
                    .build();
            
            System.out.println("Function call 후 최종 응답 생성 중...");
            String finalResponse = openAiService.createChatCompletion(followUpRequest)
                    .getChoices().get(0).getMessage().getContent();
            
            System.out.println("=== Function Call 처리 완료 ===");
            return finalResponse;
            
        } catch (Exception e) {
            System.err.println("=== Function Call 처리 오류 ===");
            System.err.println("오류 메시지: " + e.getMessage());
            e.printStackTrace();
            return "데이터 조회 중 오류가 발생했습니다: " + e.getMessage();
        }
    }
    
    /**
     * 실제 함수 실행 - 기존 StockController 코드 활용
     */
    private String executeFunction(ChatFunctionCall functionCall) {
        String functionName = functionCall.getName();
        com.fasterxml.jackson.databind.JsonNode arguments = functionCall.getArguments();
        
        try {
            switch (functionName) {
                case "get_stock_chart":
                    String stockCode = arguments.get("stock_code").asText();
                    String baseDate = arguments.has("base_date") ? arguments.get("base_date").asText() : 
                                     LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"));
                    String apiId = arguments.has("api_id") ? arguments.get("api_id").asText() : "KA10081";
                    
                    System.out.println("주식 차트 조회: " + stockCode + ", " + baseDate + ", " + apiId);
                    Map<String, Object> chartResult = kiwoomApiService.getDailyStockChart(stockCode, baseDate, apiId);
                    return summarizeChartData(chartResult);
                    
                case "get_top_volume_stocks":
                    System.out.println("거래량 상위 종목 조회");
                    Map<String, Object> volumeResult = kiwoomApiService.getDailyTopVolumeStocks();
                    return summarizeVolumeData(volumeResult);
                    
                case "get_latest_news":
                    int page = arguments.has("page") ? arguments.get("page").asInt() : 1;
                    System.out.println("최신 뉴스 조회: 페이지 " + page);
                    List<Map<String, Object>> newsResult = getLatestNewsFromDB(page);
                    return summarizeNewsList(newsResult);
                    
                case "search_news":
                    String keyword = arguments.get("keyword").asText();
                    int searchPage = arguments.has("page") ? arguments.get("page").asInt() : 1;
                    System.out.println("키워드 뉴스 검색: " + keyword + ", 페이지 " + searchPage);
                    List<Map<String, Object>> searchResult = searchNewsByKeyword(keyword, searchPage);
                    return summarizeNewsList(searchResult);
                    
                default:
                    return "알 수 없는 함수입니다: " + functionName;
            }
        } catch (Exception e) {
            System.err.println("함수 실행 오류: " + e.getMessage());
            e.printStackTrace();
            return "함수 실행 중 오류가 발생했습니다: " + e.getMessage();
        }
    }
    
    /**
     * 차트 데이터 요약 (토큰 절약)
     */
    private String summarizeChartData(Map<String, Object> chartResult) {
        if (chartResult == null || chartResult.isEmpty()) {
            return "차트 데이터를 찾을 수 없습니다.";
        }
        
        StringBuilder summary = new StringBuilder("주식 차트 데이터:\n");
        
        // 주요 정보만 추출 (전체 데이터가 아닌 핵심 요약만)
        if (chartResult.containsKey("output1")) {
            Object output1 = chartResult.get("output1");
            summary.append("기본 정보: ").append(output1.toString(), 0, Math.min(200, output1.toString().length())).append("\n");
        }
        
        if (chartResult.containsKey("output2")) {
            summary.append("최근 5일간의 차트 데이터 조회 완료\n");
        }
        
        return summary.toString();
    }
    
    /**
     * 거래량 데이터 요약 (토큰 절약)
     */
    private String summarizeVolumeData(Map<String, Object> volumeResult) {
        if (volumeResult == null || volumeResult.isEmpty()) {
            return "거래량 데이터를 찾을 수 없습니다.";
        }
        
        StringBuilder summary = new StringBuilder("거래량 상위 종목:\n");
        
        // 상위 5개 종목만 간략히 표시
        if (volumeResult.containsKey("output")) {
            Object output = volumeResult.get("output");
            String outputStr = output.toString();
            summary.append(outputStr.substring(0, Math.min(500, outputStr.length())));
            if (outputStr.length() > 500) {
                summary.append("... (더 보기)");
            }
        }
        
        return summary.toString();
    }
    
    /**
     * 뉴스 리스트 요약 (토큰 절약)
     */
    private String summarizeNewsList(List<Map<String, Object>> newsList) {
        if (newsList == null || newsList.isEmpty()) {
            return "뉴스를 찾을 수 없습니다.";
        }
        
        StringBuilder summary = new StringBuilder("뉴스 목록:\n");
        
        // 최대 5개 뉴스만 표시
        int count = 0;
        for (Map<String, Object> news : newsList) {
            if (count >= 5) break;
            
            String title = (String) news.get("title");
            String label = (String) news.get("label");
            
            // 제목이 너무 길면 자르기
            if (title != null) {
                if (title.length() > 100) {
                    title = title.substring(0, 100) + "...";
                }
                summary.append(++count).append(". [").append(label).append("] ").append(title).append("\n");
            }
        }
        
        if (newsList.size() > 5) {
            summary.append("... 외 ").append(newsList.size() - 5).append("개 뉴스");
        }
        
        return summary.toString();
    }
    
    /**
     * 기존 StockController의 getLatestNews 메서드 로직을 활용
     */
    private List<Map<String, Object>> getLatestNewsFromDB(int page) {
        List<Map<String, Object>> newsList = new ArrayList<>();
        
        try {
            int pageSize = 4;
            int offset = (page - 1) * pageSize;
            
            System.out.println("데이터베이스 연결 시도 중...");
            try (Connection conn = DriverManager.getConnection(
                    "jdbc:postgresql://db:5432/fivesense", "postgres", "1234")) {
                System.out.println("데이터베이스 연결 성공!");
                
                PreparedStatement stmt = conn.prepareStatement(
                    "SELECT title, link, label FROM news ORDER BY pub_date DESC LIMIT 40 OFFSET 0");
                System.out.println("SQL 쿼리 실행 중...");
                
                ResultSet rs = stmt.executeQuery();
                System.out.println("쿼리 실행 완료, 결과 처리 중...");
                
                int idx = 0;
                while (rs.next() && newsList.size() < 40) {
                    if (idx >= offset && newsList.size() < offset + pageSize) {
                        Map<String, Object> news = new HashMap<>();
                        news.put("title", rs.getString("title"));
                        news.put("link", rs.getString("link"));
                        news.put("label", rs.getString("label"));
                        newsList.add(news);
                    }
                    idx++;
                }
            }
        } catch (Exception e) {
            System.err.println("=== 뉴스 조회 오류 발생 ===");
            System.err.println("오류 메시지: " + e.getMessage());
            e.printStackTrace();
        }
        return newsList;
    }
    
    /**
     * 키워드로 뉴스 검색 (기존 로직 확장)
     */
    private List<Map<String, Object>> searchNewsByKeyword(String keyword, int page) {
        List<Map<String, Object>> newsList = new ArrayList<>();
        
        try {
            int pageSize = 4;
            int offset = (page - 1) * pageSize;
            
            System.out.println("데이터베이스 연결 시도 중...");
            try (Connection conn = DriverManager.getConnection(
                    "jdbc:postgresql://db:5432/fivesense", "postgres", "1234")) {
                System.out.println("데이터베이스 연결 성공!");
                
                String sql = "SELECT title, link, label FROM news WHERE title ILIKE ? OR label ILIKE ? ORDER BY pub_date DESC LIMIT ? OFFSET ?";
                PreparedStatement stmt = conn.prepareStatement(sql);
                String searchPattern = "%" + keyword + "%";
                stmt.setString(1, searchPattern);
                stmt.setString(2, searchPattern);
                stmt.setInt(3, pageSize);
                stmt.setInt(4, offset);
                
                System.out.println("SQL 쿼리 실행 중...");
                ResultSet rs = stmt.executeQuery();
                System.out.println("쿼리 실행 완료, 결과 처리 중...");
                
                while (rs.next()) {
                    Map<String, Object> news = new HashMap<>();
                    news.put("title", rs.getString("title"));
                    news.put("link", rs.getString("link"));
                    news.put("label", rs.getString("label"));
                    newsList.add(news);
                }
            }
        } catch (Exception e) {
            System.err.println("=== 키워드 뉴스 검색 오류 발생 ===");
            System.err.println("오류 메시지: " + e.getMessage());
            e.printStackTrace();
        }
        return newsList;
    }
} 