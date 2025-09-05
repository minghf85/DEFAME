# defame/evidence_retrieval/tools/text_analyzer.py

import re
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from defame.common import Action, logger
from defame.common.results import Results
from defame.evidence_retrieval.tools.tool import Tool

# spaCy imports with fallback
try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
    logger.info("spaCy loaded successfully")
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not installed, will use basic NLP functionality")


class AnalyzeText(Action):
    """Analyze text content to extract key information such as sentiment, keywords, entities, etc.
    Suitable for analyzing claim text or web content to help identify suspicious content or key information."""
    name = "analyze_text"
    
    def __init__(self, text: str, analysis_type: str = "sentiment"):
        """
        @param text: Text content to analyze
        @param analysis_type: Analysis type, options: 'sentiment' (sentiment analysis),
            'keywords' (keyword extraction), 'entities' (entity recognition), 'all' (all analyses)
        """
        self._save_parameters(locals())
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        valid_types = ['sentiment', 'keywords', 'entities', 'all']
        if analysis_type not in valid_types:
            raise ValueError(f"Analysis type must be one of: {valid_types}")
        
        self.text = text.strip()
        self.analysis_type = analysis_type
    
    def __str__(self):
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f'{self.name}(text="{text_preview}", type={self.analysis_type})'
    
    def __eq__(self, other):
        return (isinstance(other, AnalyzeText) and 
                self.text == other.text and 
                self.analysis_type == other.analysis_type)
    
    def __hash__(self):
        return hash((self.name, self.text, self.analysis_type))


@dataclass
class TextAnalysisResults(Results):
    """Text analysis results"""
    source: str
    analysis_type: str
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    keywords: Optional[List[str]] = None
    entities: Optional[Dict[str, List[str]]] = None
    confidence: Optional[float] = None
    raw_text: Optional[str] = None
    text: str = field(init=False)
    
    def __post_init__(self):
        self.text = str(self)
    
    def __str__(self):
        result_parts = [f"📊 Text Analysis Results ({self.analysis_type}):"]
        
        if self.sentiment_score is not None:
            emoji = "😊" if self.sentiment_score > 0.1 else "😞" if self.sentiment_score < -0.1 else "😐"
            result_parts.append(f"{emoji} Sentiment: {self.sentiment_label} (score: {self.sentiment_score:.2f})")
        
        if self.keywords:
            result_parts.append(f"🔑 Keywords: {', '.join(self.keywords[:8])}")
        
        if self.entities:
            for entity_type, entity_list in self.entities.items():
                if entity_list:
                    emoji_map = {
                        "dates": "📅",
                        "emails": "📧", 
                        "urls": "🔗",
                        "numbers": "🔢",
                        "phones": "📞",
                        "money": "💰"
                    }
                    emoji = emoji_map.get(entity_type, "📍")
                    result_parts.append(f"{emoji} {entity_type}: {', '.join(entity_list[:3])}")
        
        if self.confidence:
            confidence_emoji = "🎯" if self.confidence > 0.7 else "⚠️" if self.confidence > 0.4 else "❓"
            result_parts.append(f"{confidence_emoji} Confidence: {self.confidence:.2f}")
        
        return '\n'.join(result_parts)
    
    def is_useful(self) -> Optional[bool]:
        """Determine if the analysis results are useful"""
        has_results = (self.sentiment_score is not None or 
                      bool(self.keywords) or 
                      bool(self.entities))
        return has_results and (self.confidence or 0) > 0.3


class TextAnalyzer(Tool):
    """
    spaCy-based text analysis tool that can perform sentiment analysis, keyword extraction, and entity recognition.
    Suitable for analyzing claim content, news articles, social media posts, and other text information,
    helping to identify suspicious content, sentiment tendencies, and key information.
    """
    name = "text_analyzer"
    actions = [AnalyzeText]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize spaCy model
        self.nlp = None
        self._initialize_spacy()
        
        # Sentiment word dictionary for basic sentiment analysis or enhancing spaCy results
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'awesome', 'brilliant', 'perfect', 'outstanding', 'superb', 'magnificent',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'delighted',
            'true', 'correct', 'accurate', 'reliable', 'trustworthy', 'honest',
            'best', 'impressive', 'remarkable', 'exceptional', 'beautiful', 'success'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 
            'worst', 'pathetic', 'useless', 'disappointing', 'frustrating', 'annoying',
            'sad', 'angry', 'upset', 'disappointed', 'worried', 'concerned',
            'false', 'wrong', 'incorrect', 'fake', 'misleading', 'dishonest', 'lie',
            'fail', 'failure', 'problem', 'issue', 'error', 'danger', 'risk'
        }
        
        logger.info("Text analyzer tool initialized successfully")
    
    def _initialize_spacy(self):
        """Initialize spaCy model"""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available, will use basic NLP functionality")
            return
            
        try:
            # Try loading different English models
            model_names = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg', 'en']
            
            for model_name in model_names:
                try:
                    self.nlp = spacy.load(model_name)
                    logger.info(f"Successfully loaded spaCy model: {model_name}")
                    break
                except OSError:
                    continue
            
            # If no pre-trained model available, use blank English processor
            if self.nlp is None:
                self.nlp = English()
                logger.info("Using spaCy blank English processor")
                
        except Exception as e:
            logger.error(f"Failed to initialize spaCy: {e}")
            self.nlp = None
    
    def _perform(self, action: AnalyzeText) -> Results:
        logger.info(f"Performing text analysis: {action.analysis_type}")
        
        try:
            if action.analysis_type == "sentiment":
                return self._analyze_sentiment(action)
            elif action.analysis_type == "keywords":
                return self._extract_keywords(action)
            elif action.analysis_type == "entities":
                return self._extract_entities(action)
            elif action.analysis_type == "all":
                return self._analyze_all(action)
            else:
                raise ValueError(f"Unsupported analysis type: {action.analysis_type}")
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            raise
    
    def _analyze_sentiment(self, action: AnalyzeText) -> TextAnalysisResults:
        """Perform sentiment analysis using spaCy and word dictionary"""
        sentiment_score = 0.0
        confidence = 0.3
        
        if self.nlp:
            # Analyze using spaCy
            doc = self.nlp(action.text)
            
            # spaCy POS tagging and lemmatization
            positive_count = 0
            negative_count = 0
            total_words = 0
            
            for token in doc:
                if not token.is_stop and not token.is_punct and token.is_alpha:
                    total_words += 1
                    lemma = token.lemma_.lower()
                    text = token.text.lower()
                    
                    if lemma in self.positive_words or text in self.positive_words:
                        positive_count += 1
                    elif lemma in self.negative_words or text in self.negative_words:
                        negative_count += 1
            
            if total_words > 0:
                sentiment_score = (positive_count - negative_count) / total_words
                sentiment_words = positive_count + negative_count
                confidence = min(sentiment_words / total_words * 2, 1.0) if sentiment_words > 0 else 0.3
            
        else:
            # Basic method (without spaCy)
            words = re.findall(r'\b\w+\b', action.text.lower())
            positive_count = sum(1 for word in words if word in self.positive_words)
            negative_count = sum(1 for word in words if word in self.negative_words)
            
            if len(words) > 0:
                sentiment_score = (positive_count - negative_count) / len(words)
                total_sentiment_words = positive_count + negative_count
                confidence = min(total_sentiment_words / len(words) * 3, 1.0) if total_sentiment_words > 0 else 0.3
        
        # Determine sentiment label
        if sentiment_score > 0.05:
            sentiment_label = "positive"
        elif sentiment_score < -0.05:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        return TextAnalysisResults(
            source="TextAnalyzer-Sentiment",
            analysis_type="sentiment",
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            raw_text=action.text
        )
    
    def _extract_keywords(self, action: AnalyzeText) -> TextAnalysisResults:
        """Extract keywords using spaCy"""
        keywords = []
        confidence = 0.2
        
        if self.nlp:
            # Advanced keyword extraction using spaCy
            doc = self.nlp(action.text)
            
            # Word frequency statistics (using lemmatization)
            word_freq = {}
            for token in doc:
                if (not token.is_stop and 
                    not token.is_punct and 
                    not token.is_space and 
                    token.is_alpha and 
                    len(token.text) > 2):
                    
                    lemma = token.lemma_.lower()
                    word_freq[lemma] = word_freq.get(lemma, 0) + 1
            
            # Extract keywords (sorted by frequency)
            keywords = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:12]
            
            # Add named entities to keywords
            entities = [ent.text.lower() for ent in doc.ents if len(ent.text) > 2]
            keywords.extend(entities[:5])  # Add top 5 entities
            keywords = list(dict.fromkeys(keywords))  # Remove duplicates while preserving order
            
            confidence = min(len(keywords) / 8, 1.0) if keywords else 0.2
            
        else:
            # Basic method (without spaCy)
            # Simple stop words list
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
            }
            
            words = re.findall(r'\b\w+\b', action.text.lower())
            words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Word frequency statistics
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            keywords = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:12]
            confidence = min(len(keywords) / 8, 1.0) if keywords else 0.2
        
        return TextAnalysisResults(
            source="TextAnalyzer-Keywords",
            analysis_type="keywords",
            keywords=keywords,
            confidence=confidence,
            raw_text=action.text
        )
    
    def _extract_entities(self, action: AnalyzeText) -> TextAnalysisResults:
        """Extract entities using spaCy"""
        entities = {}
        confidence = 0.2
        
        if self.nlp:
            # Use spaCy's named entity recognition
            doc = self.nlp(action.text)
            
            # Organize spaCy recognized entities
            spacy_entities = {}
            for ent in doc.ents:
                label = ent.label_.lower()
                if label not in spacy_entities:
                    spacy_entities[label] = []
                spacy_entities[label].append(ent.text)
            
            # Map spaCy entity labels to our categories
            entity_mapping = {
                'person': 'persons',
                'org': 'organizations', 
                'gpe': 'locations',
                'date': 'dates',
                'time': 'dates',
                'money': 'money',
                'cardinal': 'numbers',
                'ordinal': 'numbers',
                'percent': 'percentages'
            }
            
            for spacy_label, texts in spacy_entities.items():
                mapped_label = entity_mapping.get(spacy_label, spacy_label)
                if mapped_label not in entities:
                    entities[mapped_label] = []
                entities[mapped_label].extend(texts)
        
        # Supplement with regex-based entity recognition
        regex_entities = self._extract_regex_entities(action.text)
        for entity_type, entity_list in regex_entities.items():
            if entity_type not in entities:
                entities[entity_type] = entity_list
            else:
                entities[entity_type].extend(entity_list)
        
        # Remove duplicates and limit quantity
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))[:10]
        
        # Calculate confidence
        entity_count = sum(len(v) for v in entities.values())
        if self.nlp:
            confidence = min(entity_count / 8, 1.0) if entities else 0.3
        else:
            confidence = min(entity_count / 10, 1.0) if entities else 0.2
        
        return TextAnalysisResults(
            source="TextAnalyzer-Entities",
            analysis_type="entities",
            entities=entities,
            confidence=confidence,
            raw_text=action.text
        )
    
    def _extract_regex_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using regular expressions"""
        entities = {}
        
        # Date patterns
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b'
        ]
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        if dates:
            entities["dates"] = dates
        
        # Emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            entities["emails"] = emails
        
        # URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        if urls:
            entities["urls"] = urls
        
        # Phone numbers
        phones = re.findall(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b', text)
        if phones:
            entities["phones"] = [f"{area}-{exchange}-{number}" for area, exchange, number in phones]
        
        # Numbers
        numbers = re.findall(r'\b\d{3,}\b', text)
        if numbers:
            entities["numbers"] = numbers[:10]
        
        # Money amounts
        money = re.findall(r'\$[\d,]+(?:\.\d{2})?|\d+\s*(?:dollars?|euros?|yuan|rmb)', text, re.IGNORECASE)
        if money:
            entities["money"] = money
        
        return entities
    
    def _analyze_all(self, action: AnalyzeText) -> TextAnalysisResults:
        """Comprehensive analysis (includes all types)"""
        # Perform all types of analysis
        sentiment_result = self._analyze_sentiment(action)
        keywords_result = self._extract_keywords(action)
        entities_result = self._extract_entities(action)
        
        # Safely merge confidence scores
        confidences = []
        if sentiment_result.confidence is not None:
            confidences.append(sentiment_result.confidence)
        if keywords_result.confidence is not None:
            confidences.append(keywords_result.confidence)
        if entities_result.confidence is not None:
            confidences.append(entities_result.confidence)
        
        combined_confidence = sum(confidences) / len(confidences) if confidences else 0.3
        
        return TextAnalysisResults(
            source="TextAnalyzer-Comprehensive",
            analysis_type="all",
            sentiment_score=sentiment_result.sentiment_score,
            sentiment_label=sentiment_result.sentiment_label,
            keywords=keywords_result.keywords,
            entities=entities_result.entities,
            confidence=combined_confidence,
            raw_text=action.text
        )
    
    def _summarize(self, results: TextAnalysisResults, **kwargs) -> str:
        """
        Summarize analysis results
        
        @param results: Analysis results
        @return: Summary text
        """
        summary_parts = []
        
        if results.sentiment_score is not None:
            sentiment_strength = "strong" if abs(results.sentiment_score) > 0.2 else "mild"
            summary_parts.append(f"Text shows {sentiment_strength} {results.sentiment_label} sentiment")
        
        if results.keywords:
            summary_parts.append(f"Identified {len(results.keywords)} keywords")
        
        if results.entities:
            entity_types = list(results.entities.keys())
            summary_parts.append(f"Found {len(entity_types)} entity types: {', '.join(entity_types)}")
        
        confidence_value = results.confidence or 0
        confidence_desc = "high" if confidence_value > 0.7 else "medium" if confidence_value > 0.4 else "low"
        summary_parts.append(f"Analysis confidence: {confidence_desc}")
        
        return "; ".join(summary_parts) + "."
