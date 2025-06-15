#define RELAY_PIN 9

int x;

void setup() { 
  Serial.begin(115200); 
  Serial.setTimeout(3); 
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH);
  delay(3000) // delay, milliseconds
  digitalWrite(RELAY_PIN, LOW);
}

void loop(){
	while (!Serial.available()); 
	x=Serial.readString().toInt(); 
  //Serial.write(x);
	if(x==1){
    digitalWrite(RELAY_PIN, HIGH);
    Serial.println("success: 1");
  }
  else{
    digitalWrite(RELAY_PIN, LOW);
    Serial.println("success: 0");
     //Serial.write(0);
  }
  
} 