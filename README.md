# AI

For recommendation
Do this for recommendation : docker network create my_shared_net

1. Use docker compose up --build
2. The api will run on localhost:3002
3. We have 3 apis, one is /api/delete_recommendations which has to be run when we delete the recommendations, another one is /api/add_recommendations which has to be run when we add new event to our database and final is /api/get_recommendations?eventId=1231 for recommendations
4. /api/add_recommendations: Post request, send eventId and eventData = {'event_name":ABCD, "event_description":"EFGHI"}, this will just return 200
5. /api/delete_recommendations: Post request, send only eventId
6. /api/get_recommendations?eventId=1231: Get request, send eventId in api request and this request will send you json of two things, similarity score and eventId which is same as event_id from events database.
